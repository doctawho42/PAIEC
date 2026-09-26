"""paiec/llmfeat.py without the models: prompt, truncation, keys, readout math,
plan and batches, and the shard loader, with a character-level stand-in tokenizer."""
import math
import os

import numpy as np
import pandas as pd
import pytest

from paiec import llmfeat as F
from paiec.predict import item_key

BENCHMARK_NAMES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench",
                   "swe_rebench", "mmdocrag", "swe-bench", "webarena", "gpqa", "mmlu", "aime",
                   "humaneval"]


class CharTok:
    """One token per character (ids = code points), end-of-text appended when
    add_special_tokens, and a plain chat template."""
    EOT = 0

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            return {"input_ids": [self(t, add_special_tokens)["input_ids"] for t in text]}
        ids = [ord(c) for c in text]
        return {"input_ids": ids + ([self.EOT] if add_special_tokens else [])}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return out + ("<assistant>" if add_generation_prompt else "")


def test_prompt_is_benchmark_agnostic():
    text = (F.SYSTEM + F.USER).lower()
    assert not [n for n in BENCHMARK_NAMES if n in text]
    assert F.USER.count("{task}") == 1
    for kind in ("question", "math", "software", "agent", "images"):
        assert kind in text


def test_item_text_and_keys():
    item = {"item_content": "Solve x", "item_features": "k=v", "interactors": "",
            "benchmark_id": "benchmark_123456"}
    assert F.item_text(item) == "Solve x\nk=v"
    assert F.item_text({"item_content": None}) == "\n"
    assert F.key_for(item, "benchmark_123456") == item_key(item).hex()
    other = dict(item, benchmark_id="benchmark_654321")
    assert F.key_for(item, "benchmark_654321") != F.key_for(item, "benchmark_123456")
    assert F.text_key(item) == F.text_key(other) == F.key_for(item, "")
    assert F.text_key(dict(item, item_features="k=w")) != F.text_key(item)


def test_head_tail():
    ids = list(range(20))
    assert F.head_tail(ids, 20, [-1]) == (ids, False)
    cut, tr = F.head_tail(ids, 10, [-1, -2])
    assert tr and len(cut) == 10
    assert cut == [0, 1, 2, 3, -1, -2, 16, 17, 18, 19]
    cut, _ = F.head_tail(ids, 9, [-1, -2])          # odd keep: head gets the extra
    assert cut == [0, 1, 2, 3, -1, -2, 17, 18, 19]
    with pytest.raises(ValueError):
        F.head_tail(ids, 3, [-1, -2])


def test_embedding_input_leaves_room_for_end_of_text():
    tok = CharTok()
    assert F.appended_special(tok) == [CharTok.EOT]
    ids, n, tr = F.embedding_input(tok, "x" * 5000, marker_ids=[7, 7], max_tokens=64)
    assert len(ids) == 64 and ids[-1] == CharTok.EOT and n == 5000 and tr
    ids, n, tr = F.embedding_input(tok, "abc", marker_ids=[7], max_tokens=64)
    assert ids == [97, 98, 99, CharTok.EOT] and not tr


def test_rating_prompt_span_and_mask():
    tok = CharTok()
    p = F.RatingPrompt(tok, max_task_tokens=50)
    assert p.digits == [ord(d) for d in F.DIGITS]
    ids, start, stop, mask, n, tr = p.build("short task")
    assert not tr and n == 10 and mask.all()
    assert "".join(map(chr, ids[start:stop])) == "short task"
    assert "".join(map(chr, ids)).endswith("<assistant>")
    text = "".join(chr(65 + i % 26) for i in range(500))
    ids, start, stop, mask, n, tr = p.build(text)
    assert tr and n == 500 and stop - start == 50
    span = ids[start:stop]
    marked = [t for t, keep in zip(span, mask) if not keep]
    assert "".join(map(chr, marked)) == F.MARKER
    assert "".join(map(chr, span[:5])) == text[:5] and "".join(map(chr, span[-5:])) == text[-5:]
    assert mask.sum() == 50 - len(F.MARKER)


def test_digit_stats():
    logits = np.log(np.array([0, 0, 0, 0, 0, 0, 0, 1, 3, 0], float) + 1e-300)
    lse = np.log(4 / 0.8)                  # digits hold 0.8 of the full distribution
    probs, rating, ent, mass = F.digit_stats(logits, lse)
    assert np.allclose(probs[7:9], [0.25, 0.75]) and math.isclose(probs.sum(), 1)
    assert math.isclose(rating, 7 * 0.25 + 8 * 0.75)
    assert math.isclose(ent, -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)))
    assert math.isclose(mass, 0.8)
    probs, rating, ent, mass = F.digit_stats(np.zeros((3, 10)), np.full(3, math.log(10)))
    assert np.allclose(rating, 4.5) and np.allclose(ent, math.log(10)) and np.allclose(mass, 1)


def test_mean_nll():
    assert F.mean_nll([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]) == (1.0, 3)
    assert F.mean_nll([1.0, 2.0], [2.0, 4.0], mask=[True, False]) == (1.0, 1)
    v, n = F.mean_nll([], [])
    assert n == 0 and math.isnan(v)


def test_plan_and_batches():
    lengths = [5, 3, 9, 3, 1]
    keys = ["e", "d", "c", "b", "a"]
    shards = F.plan(lengths, keys, 2)
    assert shards == [[4, 3], [1, 0], [2]]         # by length, ties by key
    assert F.plan(lengths, keys, 2) == shards
    assert F.plan(lengths, keys, 10, max_tokens=7) == [[4, 3, 1], [0], [2]]
    assert F.batches([1, 2, 2, 5, 9], budget=10, max_batch=4) == [[0, 1, 2], [3], [4]]
    assert F.batches([20], budget=10, max_batch=4) == [[0]]
    assert F.batches([1] * 10, budget=100, max_batch=4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_load_and_lookup(tmp_path):
    items = [{"item_content": f"q{i}", "item_features": "", "interactors": "",
              "benchmark_id": "b"} for i in range(3)]
    keys = [F.key_for(it, "b") for it in items]
    root = tmp_path / "b"
    (root / "emb").mkdir(parents=True)
    (root / "llm").mkdir()
    pd.DataFrame({"key": keys, "key_official": keys, "text_key": [F.text_key(i) for i in items],
                  "item_ids": [[str(i)] for i in range(3)]}).to_parquet(root / "index.parquet")
    e = np.eye(3, F.EMB_DIM, dtype=np.float16)
    np.savez(root / "emb" / "0000.npz", key=np.array(keys[:2]), emb=e[:2])
    np.savez(root / "emb" / ".0001.npz.tmp", key=np.array(keys[2:]), emb=e[2:])
    sc = {c: np.array([1.0]) for c in F.LLM_SCALARS}
    np.savez(root / "llm" / "0000.npz", key=np.array([keys[1]]),
             probs=np.full((1, 10), 0.1, np.float32), **sc)
    index, emb, probs, scalars = F.load(str(tmp_path), "b")
    assert index.has_emb.tolist() == [True, True, False]
    assert index.has_llm.tolist() == [False, True, False]
    assert np.array_equal(emb[:2], e[:2]) and np.isnan(emb[2]).all()
    assert scalars.rating.iloc[1] == 1.0 and np.isnan(scalars.rating.iloc[0])
    assert F.lookup(index, dict(items[2], benchmark_id="benchmark_999999")) == 2
    assert F.lookup(index, dict(items[2], item_content="other")) is None
