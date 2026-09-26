"""LLM features of a benchmark item's text, defined once for offline extraction
(experiments/llm_features.py) and for a run-time predictor that may reuse them.

Three features per item, all read from the item's visible text (item_text: its
item_content, a newline, its item_features):

  emb      Qwen3-Embedding-0.6B (EMB_REPO) document embedding as its model card
           prescribes for documents: no instruction prefix, the tokenizer's
           end-of-text token appended, the last token's final hidden state, L2
           normalised; stored as float16, 1024 dimensions. The text's tokens are
           cut to EMB_MAX_TOKENS - 1, end-of-text then brings them to at most
           EMB_MAX_TOKENS, keeping the head and the tail around MARKER (head_tail).
  rating   Qwen3-4B-Instruct-2507 (LLM_REPO) shown the task in one chat turn
           (SYSTEM, USER) and asked how likely a strong 2025-26 AI system is to
           solve or answer it correctly, as one digit 0..9. One forward pass, no
           generation: the next-token distribution after the assistant header,
           restricted to the ten digit tokens (digit_stats), gives `probs`
           (renormalised over the digits), the expected rating (0..9, higher is
           easier), the entropy of `probs` in nats, and `digit_mass`, the share
           of the full next-token distribution on the digits. The task's tokens
           are cut to TASK_MAX_TOKENS the same way, MARKER included.
  nll      from the same pass, the mean negative log-likelihood (nats per token)
           of the task's tokens given everything before them in the prompt,
           over the task span less the marker's tokens, and that token count.

The prompt names no benchmark and fits any task type the hidden benchmarks may
have (questions, math, code, agents, the text part of multimodal tasks). Its
exact text, the rendered chat template and the truncation rules go into the
extraction manifest.

Keys. predict.item_key digests all four visible item fields, benchmark_id
included, and a benchmark_id is not stable across views of the same item: the
raw public name in paiec.data.load_pairs, an anonymous id in paiec.official,
another anonymous id per pseudo-benchmark in paiec.testlike. So features are
stored per public benchmark under three keys, all item_key digests (hex):
`key` with the public name (the item as load_pairs builds it), `key_official`
with official's anonymous id for that name, and `text_key` with benchmark_id
blank, which matches the item under any benchmark_id (lookup uses it).

At module level this imports the standard library, numpy and paiec.predict
only (predict itself is numpy-only); the loaders import pandas where they are
used. Nothing here needs torch: the tokenizer-bound helpers take any tokenizer
object with the Hugging Face call and apply_chat_template interface.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .predict import item_key

EMB_REPO = "Qwen/Qwen3-Embedding-0.6B"
LLM_REPO = "Qwen/Qwen3-4B-Instruct-2507"
EMB_DIM = 1024
EMB_MAX_TOKENS = 2048          # end-of-text included
TASK_MAX_TOKENS = 1536         # task span of the rating prompt, marker included
MARKER = "\n[... middle of the task omitted ...]\n"
DIGITS = "0123456789"
PLACEHOLDER = "\u0000TASK\u0000"   # stands for the task while the template is rendered

SYSTEM = ("You are an expert evaluator of AI systems. You judge how hard tasks are "
          "for the strongest current AI models.")
USER = (
    "Below is one task from an AI evaluation benchmark. It could be a question to "
    "answer, a math problem, a programming or software-engineering task, a task "
    "for an agent acting on a website or in a computer environment, or the text "
    "part of a task that also involves images or documents. Metadata about the "
    "task may follow its text. Very long tasks are shortened, with the middle "
    "omitted. Benchmark tasks are chosen to challenge the best systems and are often "
    "harder than they look.\n"
    "\n"
    "<task>\n"
    "{task}\n"
    "</task>\n"
    "\n"
    "Imagine many strong AI systems of 2025-2026 each attempt this task once. What "
    "share of them solve it or answer it correctly? Most benchmark tasks are hard: the "
    "typical task is solved by fewer than half of them. Answer with one digit: 0 for "
    "under 10%, 1 for 10-20%, and so on up to 9 for 90% or more. Answer with the digit "
    "only.")
#: The question asks for the share of strong systems that succeed, in deciles,
#: with a base-rate anchor. Asked plainly ("how likely ... 0 almost certainly
#: fails, 9 almost certainly succeeds"), Qwen3-4B answered 9 for nearly every
#: public item (within-benchmark sd of the expected rating 0.01-0.08 on 6 items
#: each of the four multi-subject benchmarks, 0.42 on the web tasks); this form
#: gave 0.18-2.2 (experiments/llm_features.py, "Prompt").

#: scalar columns of an llm shard, in order
LLM_SCALARS = ("rating", "entropy", "digit_mass", "nll", "nll_tokens",
               "prompt_tokens", "task_tokens", "truncated")


def _s(v) -> str:
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def item_text(item: dict) -> str:
    """The text both models read: item_content, a newline, item_features."""
    return f"{_s(item.get('item_content'))}\n{_s(item.get('item_features'))}"


def text_key(item: dict) -> str:
    """predict.item_key of the item with benchmark_id blank, as hex: the same for
    every benchmark_id the item may carry."""
    return item_key({**item, "benchmark_id": ""}).hex()


def key_for(item: dict, benchmark_id: str) -> str:
    """predict.item_key of the item under `benchmark_id`, as hex."""
    return item_key({**item, "benchmark_id": benchmark_id}).hex()


def head_tail(ids, max_len: int, marker=()) -> tuple[list, bool]:
    """`ids` if they fit in max_len, else their head and tail around `marker`, in
    exactly max_len tokens (the head gets the odd one). Returns (ids, truncated)."""
    ids, marker = list(ids), list(marker)
    if len(ids) <= max_len:
        return ids, False
    keep = max_len - len(marker)
    if keep < 2:
        raise ValueError(f"max_len {max_len} leaves no room around a {len(marker)}-token marker")
    head = (keep + 1) // 2
    return ids[:head] + marker + ids[len(ids) - (keep - head):], True


def digit_stats(digit_logits, lse):
    """From the ten digit logits at the answer position and the log-sum-exp of
    all logits there: (probs over the digits, expected rating, entropy in nats,
    digit mass). Works on any leading shape; computed in float64."""
    x = np.asarray(digit_logits, np.float64)
    lse = np.asarray(lse, np.float64)
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    z = e.sum(-1, keepdims=True)
    probs = e / z
    expected = probs @ np.arange(10, dtype=np.float64)
    plogp = np.where(probs > 0, probs * np.log(np.where(probs > 0, probs, 1.0)), 0.0)
    entropy = -plogp.sum(-1)
    mass = np.exp(m[..., 0] + np.log(z[..., 0]) - lse)
    return probs, expected, entropy, mass


def mean_nll(target_logits, lse, mask=None):
    """Mean of lse - target logit over the positions `mask` keeps, and their
    count; (nan, 0) when none are kept."""
    nll = np.asarray(lse, np.float64) - np.asarray(target_logits, np.float64)
    if mask is not None:
        nll = nll[np.asarray(mask, bool)]
    return (float(nll.mean()) if nll.size else float("nan")), int(nll.size)


# --- tokenizer-bound pieces ---------------------------------------------------------

def encode(tok, text: str) -> list:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def appended_special(tok) -> list:
    """The token ids the tokenizer appends to a text by default (for the
    embedding model's tokenizer: end-of-text)."""
    plain = encode(tok, "a")
    full = list(tok("a")["input_ids"])
    if full[:len(plain)] != plain:
        raise ValueError("tokenizer prepends special tokens; last-token pooling expects "
                         "only appended ones")
    return full[len(plain):]


def embedding_input(tok, text: str, marker_ids=None, max_tokens=EMB_MAX_TOKENS):
    """Token ids for the embedding model: the text cut by head_tail to leave room
    for the appended end-of-text, then end-of-text. Returns (ids, n_text_tokens
    before truncation, truncated)."""
    marker_ids = encode(tok, MARKER) if marker_ids is None else marker_ids
    return embedding_ids(encode(tok, text), appended_special(tok), marker_ids, max_tokens)


def embedding_ids(ids, tail, marker_ids, max_tokens=EMB_MAX_TOKENS):
    """embedding_input on the text's ids, given the appended ids (`tail`) and the
    marker's."""
    ids = list(ids)
    cut, truncated = head_tail(ids, max_tokens - len(tail), marker_ids)
    return cut + list(tail), len(ids), truncated


def digit_token_ids(tok) -> list:
    out = []
    for d in DIGITS:
        ids = encode(tok, d)
        if len(ids) != 1:
            raise ValueError(f"digit {d!r} is {len(ids)} tokens in this tokenizer")
        out.append(ids[0])
    return out


class RatingPrompt:
    """The rating prompt bound to one tokenizer. The chat template is rendered
    once around PLACEHOLDER; the text before and after it, the marker and the
    task are tokenized separately and joined as ids, so the task span's position
    is known exactly."""

    def __init__(self, tok, max_task_tokens=TASK_MAX_TOKENS):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.replace("{task}", PLACEHOLDER)}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text.count(PLACEHOLDER) != 1:
            raise ValueError("chat template did not keep the task placeholder exactly once")
        self.rendered = text
        pre, post = text.split(PLACEHOLDER)
        self.prefix, self.suffix = encode(tok, pre), encode(tok, post)
        self.marker = encode(tok, MARKER)
        self.digits = digit_token_ids(tok)
        self.max_task_tokens = max_task_tokens
        self._tok = tok

    def build(self, text: str):
        """(ids, span start, span stop, score mask over the span, task tokens
        before truncation, truncated). The mask drops the marker's tokens."""
        return self.build_ids(encode(self._tok, text))

    def build_ids(self, ids):
        """build on the task text's ids (encode(tok, text))."""
        ids = list(ids)
        cut, truncated = head_tail(ids, self.max_task_tokens, self.marker)
        mask = np.ones(len(cut), bool)
        if truncated:
            head = (self.max_task_tokens - len(self.marker) + 1) // 2
            mask[head:head + len(self.marker)] = False
        start = len(self.prefix)
        return (self.prefix + cut + self.suffix, start, start + len(cut), mask, len(ids),
                truncated)

    def describe(self) -> dict:
        return {"system": SYSTEM, "user": USER, "placeholder": PLACEHOLDER,
                "rendered": self.rendered, "marker": MARKER,
                "max_task_tokens": self.max_task_tokens,
                "prefix_tokens": len(self.prefix), "suffix_tokens": len(self.suffix),
                "marker_tokens": len(self.marker), "digit_token_ids": self.digits}


# --- work plan ----------------------------------------------------------------------

def plan(lengths, keys, size: int, max_tokens: int | None = None) -> list:
    """Shards of at most `size` item indices and, if given, `max_tokens` tokens
    (a longer item gets a shard of its own), items ordered by (length, key): a
    pure function of the items, so a resumed run cuts the same shards."""
    order = sorted(range(len(keys)), key=lambda i: (int(lengths[i]), keys[i]))
    out, cur, tok = [], [], 0
    for i in order:
        n = int(lengths[i])
        if cur and (len(cur) >= size or (max_tokens is not None and tok + n > max_tokens)):
            out.append(cur)
            cur, tok = [], 0
        cur.append(i)
        tok += n
    if cur:
        out.append(cur)
    return out


def batches(lengths, budget: int, max_batch: int) -> list:
    """Consecutive runs of positions 0..n-1 whose padded size (count x longest)
    stays within `budget` tokens and `max_batch` items; one item always fits."""
    out, cur, longest = [], [], 0
    for i, n in enumerate(lengths):
        L = max(longest, int(n))
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * L > budget):
            out.append(cur)
            cur, L = [], int(n)
        cur.append(i)
        longest = L
    if cur:
        out.append(cur)
    return out


def digest(obj) -> str:
    """sha256 of an object's canonical JSON (the plan and config fingerprints)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


# --- reading extracted features -----------------------------------------------------

def load(out_dir: str, benchmark: str):
    """The features of one public benchmark as (index, emb, probs, scalars):
    index a DataFrame (key, key_official, text_key, item_ids, ...), emb float16
    [n, EMB_DIM] and probs float32 [n, 10] aligned with it, scalars a DataFrame
    of LLM_SCALARS. Rows whose shard is not written yet are NaN (has_emb,
    has_llm say which)."""
    import pandas as pd
    root = os.path.join(out_dir, benchmark)
    index = pd.read_parquet(os.path.join(root, "index.parquet"))
    pos = {k: i for i, k in enumerate(index["key"])}
    n = len(index)
    emb = np.full((n, EMB_DIM), np.nan, np.float16)
    probs = np.full((n, 10), np.nan, np.float32)
    scalars = {c: np.full(n, np.nan) for c in LLM_SCALARS}
    has_emb, has_llm = np.zeros(n, bool), np.zeros(n, bool)
    for kind in ("emb", "llm"):
        d = os.path.join(root, kind)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz") or f.startswith("."):     # .NNNN.npz.tmp: unfinished
                continue
            with np.load(os.path.join(d, f)) as z:
                rows = np.array([pos[k] for k in z["key"]], np.int64)
                if kind == "emb":
                    emb[rows] = z["emb"]
                    has_emb[rows] = True
                else:
                    probs[rows] = z["probs"]
                    for c in LLM_SCALARS:
                        scalars[c][rows] = z[c]
                    has_llm[rows] = True
    index = index.assign(has_emb=has_emb, has_llm=has_llm)
    return index, emb, probs, pd.DataFrame(scalars, index=index.index)


def lookup(index, item: dict):
    """Row position of `item` in a loaded index by text_key, or None."""
    hits = np.flatnonzero(index["text_key"].to_numpy() == text_key(item))
    return int(hits[0]) if len(hits) else None
