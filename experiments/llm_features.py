"""Per-item LLM features of the public binary benchmarks, into data/features/.

For every unique item of matharena, multi_swebench, real_webagents,
researchcodebench and then swe_rebench (mmdocrag is not eligible), computes the
features paiec/llmfeat.py defines: a Qwen3-Embedding-0.6B embedding of the item
text, and from one forward pass of Qwen3-4B-Instruct-2507 over the rating prompt
the digit distribution (expected rating, probabilities, entropy, digit mass) and
the mean NLL of the task text. Nothing is generated or sampled.

Items are read from data/<benchmark>/items.parquet as paiec.data.load_pairs
builds them and deduplicated on paiec.predict.item_key (matharena has 200 items
whose text repeats another's); each unique item keeps every item_id it stands
for. Work runs in four stages, so the four multi-subject benchmarks are done
before swe_rebench (one subject) starts: embeddings of the four, ratings of the
four, embeddings of swe_rebench, ratings of swe_rebench. Each model is loaded for
its stage and freed after it.

Output (DIR = data/features by default):
  DIR/manifest.json              models and their commit hashes, prompt text and
                                 the rendered chat template, truncation rules,
                                 dtypes, batch budgets, the shard plan's hash,
                                 per-benchmark progress, environment
  DIR/<benchmark>/index.parquet  one row per unique item: key, key_official,
                                 text_key (llmfeat), item_ids, n_chars, token
                                 counts, shard numbers
  DIR/<benchmark>/emb/NNNN.npz   key, emb (float16), tokens, text_tokens, truncated
  DIR/<benchmark>/llm/NNNN.npz   key, probs, llmfeat.LLM_SCALARS
Read them back with paiec.llmfeat.load(DIR, benchmark).

Resuming. Shards are fixed in advance (llmfeat.plan: items ordered by token
length, then key), and batches inside a shard depend on the shard alone, so a
resumed run computes exactly what an uninterrupted one would. Every shard is
written atomically; --resume skips the ones on disk after checking that the
models, prompt, dtypes, budgets and plan are the ones in the manifest (else it
refuses). Without --resume an existing manifest is an error.

Prompt. The rating question went through two rounds on 24 public items (six
per multi-subject benchmark, fp16): asked plainly how likely a strong system is
to succeed (0 fails ... 9 succeeds), with or without decile anchors, the model
put nearly all mass on 9 (within-benchmark sd of the expected rating 0.01-0.42);
asked how likely it is to fail, it put it on 7-8, so the 4B model leans to high
digits whatever they mean. The shipped form (llmfeat.USER) asks for the share
of strong systems that succeed, in deciles, with the anchor that the typical
task is solved by fewer than half: sd 0.55, 0.18, 2.2, 0.52 on matharena,
multi_swebench, real_webagents, researchcodebench. Six items say nothing about
correlation with difficulty; the choice is on spread alone.

Precision. On 12 items the fp16 run differs from fp32 by at most 0.011 in the
expected rating, 0.011 in a digit probability and 0.003 nats in the NLL (bf16:
0.024, 0.023, 0.022), so both models run in fp16; the output projection is
computed in fp32. StreamedLM matches the whole transformers model exactly (max
difference 0 at real positions, fp32 and fp16; --check-streamed).

Determinism: no sampling anywhere, fixed batches, fixed model revisions (read
from the local snapshot path; nothing is downloaded unless --download).

Usage:
  python experiments/llm_features.py --download          # fetch both models once
  python experiments/llm_features.py --validate 50 --out /tmp/x   # smoke test
  nohup python experiments/llm_features.py --resume > data/features/extract.out 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import sys


def _threads(argv) -> int:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return 4


# before numpy, torch or tokenizers read them
_T = str(_threads(sys.argv))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = _T
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import gc                   # noqa: E402
import glob                 # noqa: E402
import json                 # noqa: E402
import platform             # noqa: E402
import subprocess           # noqa: E402
import time                 # noqa: E402
from datetime import datetime, timedelta   # noqa: E402

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paiec import data as D          # noqa: E402
from paiec import llmfeat as F       # noqa: E402

FIRST = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
LAST = ["swe_rebench"]
STAGES = [("emb", FIRST), ("llm", FIRST), ("emb", LAST), ("llm", LAST)]
OUT = os.path.join(D.DATA_DIR, "features")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _dur(s):
    if s is None or not np.isfinite(s):
        return "?"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def atomic_write(path, write):
    tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, lambda fh: fh.write(json.dumps(obj, indent=1, sort_keys=True).encode()))


# --- items and plan -----------------------------------------------------------------

def unique_items(benchmark):
    """Unique items of a benchmark as load_pairs builds them, sorted by key."""
    it = pd.read_parquet(os.path.join(D.DATA_DIR, benchmark, "items.parquet"),
                         columns=["item_id", "benchmark_id", "content", "item_features"])
    anon = D.anon_id("benchmark", benchmark)
    by = {}
    for iid, bid, content, feats in zip(it.item_id, it.benchmark_id, it.content,
                                        it.item_features):
        item = {"item_content": D._clean(content), "item_features": D._clean(feats),
                "interactors": "", "benchmark_id": D._clean(bid)}
        k = F.key_for(item, item["benchmark_id"])
        if k not in by:
            by[k] = {"key": k, "key_official": F.key_for(item, anon),
                     "text_key": F.text_key(item), "item": item, "item_ids": []}
        by[k]["item_ids"].append(str(iid))
    rows = [by[k] for k in sorted(by)]
    for r in rows:
        r["item_ids"] = sorted(r["item_ids"])
    return rows, len(it)


class Inputs:
    """Token ids of every item for both models, and the shard plan."""

    def __init__(self, benchmarks, emb_tok, llm_tok, emb_shard, llm_shard, llm_shard_tokens,
                 keep=None):
        self.prompt = F.RatingPrompt(llm_tok)
        marker = F.encode(emb_tok, F.MARKER)
        tail = F.appended_special(emb_tok)
        self.by = {}
        for b in benchmarks:
            rows, n_rows = unique_items(b)
            if keep is not None:
                rows = [r for r in rows if r["key"] in keep.get(b, ())]
            emb, llm = [], []
            texts = [F.item_text(r["item"]) for r in rows]
            e_ids = emb_tok(texts, add_special_tokens=False)["input_ids"] if texts else []
            l_ids = llm_tok(texts, add_special_tokens=False)["input_ids"] if texts else []
            for r, text, e, l in zip(rows, texts, e_ids, l_ids):
                r["n_chars"] = len(text)
                ids, n_text, trunc = F.embedding_ids(e, tail, marker)
                r.update(emb_tokens=len(ids), emb_text_tokens=n_text, emb_truncated=trunc)
                emb.append(np.array(ids, np.int32))
                ids, start, stop, mask, n_task, trunc = self.prompt.build_ids(l)
                r.update(llm_tokens=len(ids), task_tokens=n_task, task_truncated=trunc)
                llm.append((np.array(ids, np.int32), start, stop, mask))
            keys = [r["key"] for r in rows]
            eplan = F.plan([len(x) for x in emb], keys, emb_shard)
            lplan = F.plan([len(x[0]) for x in llm], keys, llm_shard, llm_shard_tokens)
            for s, shard in enumerate(eplan):
                for i in shard:
                    rows[i]["emb_shard"] = s
            for s, shard in enumerate(lplan):
                for i in shard:
                    rows[i]["llm_shard"] = s
            self.by[b] = {"rows": rows, "n_rows": n_rows, "emb": emb, "llm": llm,
                          "plan": {"emb": eplan, "llm": lplan}}

    def index(self, b):
        cols = ["key", "key_official", "text_key", "item_ids", "n_chars", "emb_tokens",
                "emb_text_tokens", "emb_truncated", "llm_tokens", "task_tokens",
                "task_truncated", "emb_shard", "llm_shard"]
        return pd.DataFrame([{c: r[c] for c in cols} for r in self.by[b]["rows"]])

    def plan_fingerprint(self):
        return F.digest({b: {k: [[v["rows"][i]["key"] for i in s] for s in v["plan"][k]]
                             for k in ("emb", "llm")} for b, v in self.by.items()})

    def tokens(self, b, kind, shard=None):
        v = self.by[b]
        idx = range(len(v["rows"])) if shard is None else v["plan"][kind][shard]
        return int(sum(len(v[kind][i]) if kind == "emb" else len(v[kind][i][0]) for i in idx))


# --- models -------------------------------------------------------------------------

def snapshot(repo, download=False):
    """Local snapshot path and commit hash of a model repo; no network unless
    `download`."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo, local_files_only=not download,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE",
                                             "README.md"])
    return path, os.path.basename(os.path.normpath(path))


def load_weights(model_cls, path, dtype, dev, **kw):
    """Build the model directly on `dev` with uninitialised weights and copy the
    checkpoint in one tensor at a time: no second full copy in memory (loading
    on the CPU and moving would briefly hold two, 16 GB for the 4B model)."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights
    cfg = AutoConfig.from_pretrained(path)
    _no_sdpa_gqa()
    with no_init_weights(), torch.device(dev):
        model = model_cls.from_config(cfg, dtype=dtype, attn_implementation="sdpa", **kw)
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded = set()
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, framework="pt", device="cpu") as fh:
                for name in fh.keys():
                    target = name if name in params else ("model." + name
                                                          if "model." + name in params else None)
                    if target is None:
                        if name == "lm_head.weight":
                            continue
                        raise KeyError(f"{name} of {f} has no place in {model_cls.__name__}")
                    params[target].copy_(fh.get_tensor(name))
                    loaded.add(target)
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
        loaded.update(n for n in params if n.endswith("lm_head.weight"))
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"weights missing from the checkpoint: {missing[:5]}")
    for n, p in params.items():
        if not torch.isfinite(p).all():
            raise RuntimeError(f"non-finite weight {n} after conversion to {dtype}")
    return model.eval()


def _no_sdpa_gqa():
    """torch 2.6's MPS scaled_dot_product_attention fails on grouped-query
    attention (enable_gqa) for long unpadded batches ("incompatible dimensions"),
    which transformers requests whenever a batch has no padding. Make it repeat the
    key/value heads instead, as it does for padded batches: same numbers, a
    little more memory."""
    from transformers.integrations import sdpa_attention
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False


def _pad(seqs, pad_id):
    import torch
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.from_numpy(s.astype(np.int64))
        mask[r, :len(s)] = 1
    return ids, mask


class Embedder:
    def __init__(self, path, dtype, dev, pad_id, budget, max_batch):
        import torch
        from transformers import AutoModel
        self.torch, self.dev, self.pad_id = torch, dev, pad_id
        self.budget, self.max_batch = budget, max_batch
        self.model = load_weights(AutoModel, path, dtype, dev)

    def shard(self, seqs):
        torch = self.torch
        out = np.zeros((len(seqs), F.EMB_DIM), np.float16)
        for b in F.batches([len(s) for s in seqs], self.budget, self.max_batch):
            ids, mask = _pad([seqs[i] for i in b], self.pad_id)
            with torch.inference_mode():
                hs = self.model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev),
                                use_cache=False).last_hidden_state
                last = hs[torch.arange(len(b), device=self.dev),
                          (mask.sum(1) - 1).to(self.dev)].float()
                if not torch.isfinite(last).all():
                    raise FloatingPointError("non-finite embedding")
                e = torch.nn.functional.normalize(last, p=2, dim=-1)
            out[b] = e.cpu().numpy().astype(np.float16)
        return out


class StreamedLM:
    """A Qwen3 decoder run one layer at a time over a whole shard.

    The 4B model in half precision is 8 GB; on a 16 GB machine shared with other
    jobs, holding it on the GPU swapped it in and out on every forward (15 tokens
    a second). Here only the token embedding (also the output projection: the
    weights are tied), one decoder layer and the shard's hidden states are
    resident, about 2.5 GB. For each layer in turn its weights are read from the
    checkpoint (bf16) into one reused transformers Qwen3DecoderLayer in `dtype`,
    and every batch of the shard goes through it; reading the 36 layers costs a
    few seconds a shard, and the arithmetic is the same kernels on the same
    batches as the whole model's forward. Attention is plain causal (no mask):
    batches are right-padded, so no real token ever sees a pad, and a pad's own
    garbage is never read. check() compares this against the whole model."""

    def __init__(self, path, dtype, dev):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.modeling_utils import no_init_weights
        from transformers.models.qwen3 import modeling_qwen3 as Q
        _no_sdpa_gqa()
        self.torch, self.dev, self.dtype = torch, dev, dtype
        cfg = AutoConfig.from_pretrained(path)
        cfg._attn_implementation = "sdpa"
        if getattr(cfg, "use_sliding_window", False):
            raise ValueError("sliding-window layers are not supported")
        self.cfg = cfg
        self.files = {}
        self.handles = [safe_open(f, framework="pt", device="cpu")
                        for f in sorted(glob.glob(os.path.join(path, "*.safetensors")))]
        for h in self.handles:
            for name in h.keys():
                self.files[name] = h
        self.prefix = "model." if "model.embed_tokens.weight" in self.files else ""
        self.embed = self._tensor(self.prefix + "embed_tokens.weight")
        head = "lm_head.weight"
        self.head = self._tensor(head) if head in self.files else self.embed
        with no_init_weights(), torch.device(dev):
            self.norm = Q.Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dtype)
            self.layer = Q.Qwen3DecoderLayer(cfg, 0).to(dtype).eval()
            self.rotary = Q.Qwen3RotaryEmbedding(cfg)
        with torch.no_grad():
            self.norm.weight.copy_(self._tensor(self.prefix + "norm.weight"))
        self.params = dict(self.layer.named_parameters())
        self.n_layers = cfg.num_hidden_layers

    def _tensor(self, name):
        t = self.files[name].get_tensor(name).to(self.dtype).to(self.dev)
        if not self.torch.isfinite(t).all():
            raise RuntimeError(f"non-finite weight {name} in {self.dtype}")
        return t

    def _load_layer(self, i):
        torch = self.torch
        pre = f"{self.prefix}layers.{i}."
        with torch.no_grad():
            for n, prm in self.params.items():
                prm.copy_(self.files[pre + n].get_tensor(pre + n))

    def forward(self, batches):
        """batches: (ids [B, L] long) on any device. Returns the final-norm hidden
        states [B, L, H] per batch, on the device."""
        torch = self.torch
        with torch.inference_mode():
            hs = [torch.nn.functional.embedding(ids.to(self.dev), self.embed) for ids in batches]
            pos = [torch.arange(h.shape[1], device=self.dev)[None] for h in hs]
            pe = [self.rotary(h, p) for h, p in zip(hs, pos)]
            for i in range(self.n_layers):
                self._load_layer(i)
                for j in range(len(hs)):
                    out = self.layer(hs[j], attention_mask=None, position_ids=pos[j],
                                     position_embeddings=pe[j], use_cache=False)
                    hs[j] = out[0] if isinstance(out, tuple) else out
            return [self.norm(h) for h in hs]


def check_streamed(path, dtype, dev, seqs, pad_id):
    """Largest absolute difference between StreamedLM and the whole transformers
    model on the same right-padded batch, at real positions (run on the small
    embedding model, which has the same Qwen3 layers and fits in memory)."""
    import torch
    from transformers import AutoModel
    ids, mask = _pad(seqs, pad_id)
    full = load_weights(AutoModel, path, dtype, dev)
    with torch.inference_mode():
        ref = full(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                   use_cache=False).last_hidden_state.float().cpu()
    del full
    got = StreamedLM(path, dtype, dev).forward([ids])[0].float().cpu()
    real = mask.bool()
    return float((ref - got).abs()[real].max()), float(ref.abs()[real].max())


class Rater:
    """One forward pass per prompt (StreamedLM, shard by shard); the final
    projection in float32 over vocabulary chunks (online log-sum-exp), so the
    digit distribution and the NLL do not inherit half-precision logits."""

    def __init__(self, path, dtype, dev, pad_id, digits, budget, max_batch, vocab_chunk=16384):
        import torch
        self.torch, self.dev, self.pad_id = torch, dev, pad_id
        self.budget, self.max_batch, self.chunk = budget, max_batch, vocab_chunk
        self.lm = StreamedLM(path, dtype, dev)
        self.W = self.lm.head
        self.digits = torch.tensor(digits, device=dev)

    def _lse(self, h):
        torch = self.torch
        m = torch.full((h.shape[0],), -float("inf"), device=self.dev)
        s = torch.zeros(h.shape[0], device=self.dev)
        for v0 in range(0, self.W.shape[0], self.chunk):
            lg = h @ self.W[v0:v0 + self.chunk].float().T
            nm = torch.maximum(m, lg.amax(1))
            s = s * torch.exp(m - nm) + torch.exp(lg - nm[:, None]).sum(1)
            m = nm
        return m + torch.log(s)

    def shard(self, items):
        """items: (ids, span start, span stop, mask) per prompt."""
        torch = self.torch
        n = len(items)
        res = {c: np.zeros(n) for c in F.LLM_SCALARS}
        res["probs"] = np.zeros((n, 10), np.float32)
        groups = F.batches([len(x[0]) for x in items], self.budget, self.max_batch)
        hidden = self.lm.forward([_pad([items[i][0] for i in b], self.pad_id)[0]
                                  for b in groups])
        for b, hs in zip(groups, hidden):
            L = hs.shape[1]
            rows, tgts, owner, last = [], [], [], []
            for r, i in enumerate(b):
                seq, start, stop, keep = items[i]
                pos = np.arange(start, stop)[keep]
                rows.append(r * L + pos - 1)          # hidden state that predicts pos
                tgts.append(seq[pos])
                owner.append(np.full(len(pos), r))
                last.append(r * L + len(seq) - 1)
            rows = torch.from_numpy(np.concatenate(rows)).to(self.dev)
            tgts = torch.from_numpy(np.concatenate(tgts).astype(np.int64)).to(self.dev)
            owner = np.concatenate(owner)
            last = torch.tensor(last, device=self.dev)
            with torch.inference_mode():
                flat = hs.reshape(-1, hs.shape[-1])
                h_last = flat[last].float()
                if not torch.isfinite(h_last).all():
                    raise FloatingPointError("non-finite hidden state")
                dig = h_last @ self.W[self.digits].float().T
                lse_last = self._lse(h_last)
                tl, tlse = [], []
                for p0 in range(0, len(rows), 2048):
                    h = flat[rows[p0:p0 + 2048]].float()
                    tl.append((h * self.W[tgts[p0:p0 + 2048]].float()).sum(1))
                    tlse.append(self._lse(h))
                tl = torch.cat(tl).cpu().numpy() if tl else np.zeros(0)
                tlse = torch.cat(tlse).cpu().numpy() if tlse else np.zeros(0)
            if not (np.isfinite(tl).all() and np.isfinite(tlse).all()):
                raise FloatingPointError("non-finite log-likelihood")
            probs, expected, entropy, mass = F.digit_stats(dig.cpu().numpy(),
                                                           lse_last.cpu().numpy())
            for r, i in enumerate(b):
                sel = owner == r
                nll, cnt = F.mean_nll(tl[sel], tlse[sel])
                res["probs"][i] = probs[r]
                res["rating"][i], res["entropy"][i] = expected[r], entropy[r]
                res["digit_mass"][i], res["nll"][i], res["nll_tokens"][i] = mass[r], nll, cnt
                res["prompt_tokens"][i] = len(items[i][0])
        return res


# --- the run ------------------------------------------------------------------------

def git_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except Exception:
        return None


def validation_keys(benchmarks, n, rows_of):
    """About n items spread over the benchmarks, the first by blake2b of the key."""
    import hashlib
    out, per = {}, [n // len(benchmarks) + (i < n % len(benchmarks))
                    for i in range(len(benchmarks))]
    for b, k in zip(benchmarks, per):
        keys = sorted((r["key"] for r in rows_of(b)),
                      key=lambda x: hashlib.blake2b(x.encode(), digest_size=8).digest())
        out[b] = set(keys[:k])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--download", action="store_true",
                    help="fetch the two models (the only network use) and exit")
    ap.add_argument("--validate", type=int, default=0,
                    help="smoke test on about this many items of the four multi-subject "
                         "benchmarks; prints sanity checks and a runtime estimate")
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help="restrict the stages to these benchmarks")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--emb-dtype", default="float16")
    ap.add_argument("--llm-dtype", default="float16")
    ap.add_argument("--emb-shard", type=int, default=256)
    ap.add_argument("--llm-shard", type=int, default=256)
    ap.add_argument("--llm-shard-tokens", type=int, default=65536,
                    help="token cap of an llm shard (the hidden states held per shard)")
    ap.add_argument("--check-streamed", action="store_true",
                    help="compare StreamedLM with the whole model on the embedding model "
                         "and exit")
    ap.add_argument("--emb-budget", type=int, default=8192, help="padded tokens per batch")
    ap.add_argument("--llm-budget", type=int, default=4096, help="padded tokens per batch")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--rates", type=float, nargs=2, metavar=("EMB", "LLM"),
                    help="tokens/s to assume for the ETA until measured (from --validate)")
    args = ap.parse_args(argv)

    if args.download:
        for repo in (F.EMB_REPO, F.LLM_REPO):
            path, rev = snapshot(repo, download=True)
            log(repo, rev, path)
        return

    import torch
    import transformers
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    emb_path, emb_rev = snapshot(F.EMB_REPO)
    llm_path, llm_rev = snapshot(F.LLM_REPO)
    emb_tok = AutoTokenizer.from_pretrained(emb_path)
    llm_tok = AutoTokenizer.from_pretrained(llm_path)

    if args.check_streamed:
        texts = [F.item_text(r["item"]) for r in unique_items("matharena")[0][:6]]
        seqs = [np.array(F.encode(emb_tok, t)[:300 + 60 * i], np.int32)
                for i, t in enumerate(texts)]
        for name in ("float32", args.llm_dtype):
            diff, scale = check_streamed(emb_path, dtype[name], args.device, seqs,
                                         emb_tok.pad_token_id)
            log(f"StreamedLM vs whole model, {name}: max |diff| {diff:.3g} "
                f"(max |h| {scale:.3g})")
        return

    stages = [(k, [b for b in bs if not args.benchmarks or b in args.benchmarks])
              for k, bs in STAGES]
    stages = [(k, bs) for k, bs in stages if bs]
    every = list(dict.fromkeys(b for _, bs in stages for b in bs))
    if args.validate:
        stages = [(k, bs) for k, bs in stages if bs[0] in FIRST]
    benchmarks = list(dict.fromkeys(b for _, bs in stages for b in bs))

    t0 = time.time()
    full = Inputs(every, emb_tok, llm_tok, args.emb_shard, args.llm_shard,
                  args.llm_shard_tokens)
    inp = full
    if args.validate:
        keep = validation_keys(benchmarks, args.validate,
                               lambda b: full.by[b]["rows"])
        inp = Inputs(benchmarks, emb_tok, llm_tok, args.emb_shard, args.llm_shard,
                     args.llm_shard_tokens, keep)
    log(f"tokenized {sum(len(v['rows']) for v in inp.by.values())} unique items "
        f"in {time.time() - t0:.0f}s")

    config = {
        "item_text": "item_content + '\\n' + item_features",
        "embedding": {"repo": F.EMB_REPO, "revision": emb_rev, "max_tokens": F.EMB_MAX_TOKENS,
                      "appended": F.appended_special(emb_tok), "marker": F.MARKER,
                      "truncation": "head_tail: text tokens cut to max_tokens - 1, head gets "
                                    "the odd token, marker between head and tail",
                      "pooling": "last token (the appended end-of-text), right padding",
                      "normalize": "L2 in float32", "stored": "float16",
                      "dtype": args.emb_dtype, "budget": args.emb_budget,
                      "shard": args.emb_shard},
        "rating": {"repo": F.LLM_REPO, "revision": llm_rev, **inp.prompt.describe(),
                   "truncation": "head_tail on the task's tokens to max_task_tokens, marker "
                                 "included; head gets the odd token",
                   "readout": "next-token logits after the assistant header; final "
                              "projection in float32; probs renormalised over the digit "
                              "tokens; rating = sum d * p_d; entropy in nats; digit_mass = "
                              "share of the full softmax on the digits",
                   "nll": "mean over the task span less the marker tokens of -log p(token | "
                          "prompt before it), nats",
                   "dtype": args.llm_dtype, "budget": args.llm_budget,
                   "shard": args.llm_shard, "shard_tokens": args.llm_shard_tokens,
                   "runner": "StreamedLM: one decoder layer at a time over a shard, "
                             "causal attention without a mask on right-padded batches"},
        "max_batch": args.max_batch, "device": args.device,
        "validate": args.validate or None,
        "benchmarks": benchmarks,
    }
    config_hash = F.digest(config)
    plan_hash = inp.plan_fingerprint()

    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "manifest.json")
    if os.path.exists(mpath):
        if not args.resume:
            sys.exit(f"{mpath} exists: pass --resume to continue it, or choose another --out")
        with open(mpath) as fh:
            manifest = json.load(fh)
        if manifest.get("config_hash") != config_hash or manifest.get("plan_hash") != plan_hash:
            sys.exit("the manifest was written for other models, prompt, dtypes, budgets or "
                     "items; refusing to mix outputs (use another --out)")
        log(f"resuming {args.out}")
    else:
        manifest = {"config": config, "config_hash": config_hash, "plan_hash": plan_hash,
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "env": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__,
                            "numpy": np.__version__, "machine": platform.platform(),
                            "git_commit": git_commit(), "argv": sys.argv},
                    "stages": [[k, bs] for k, bs in stages]}
    manifest["benchmarks"] = {
        b: {"rows": v["n_rows"], "unique": len(v["rows"]),
            "emb_shards": len(v["plan"]["emb"]), "llm_shards": len(v["plan"]["llm"]),
            "emb_tokens": inp.tokens(b, "emb"), "llm_tokens": inp.tokens(b, "llm"),
            "emb_truncated": int(sum(r["emb_truncated"] for r in v["rows"])),
            "task_truncated": int(sum(r["task_truncated"] for r in v["rows"]))}
        for b, v in inp.by.items()}

    for b in benchmarks:
        root = os.path.join(args.out, b)
        for k in ("emb", "llm"):
            os.makedirs(os.path.join(root, k), exist_ok=True)
        idx = inp.index(b)
        ipath = os.path.join(root, "index.parquet")
        if os.path.exists(ipath):
            old = pd.read_parquet(ipath)
            if list(old["key"]) != list(idx["key"]):
                sys.exit(f"{ipath} holds other items; refusing to mix outputs")
        else:
            atomic_write(ipath, lambda fh: idx.to_parquet(fh, index=False))

    def shard_path(b, kind, s):
        return os.path.join(args.out, b, kind, f"{s:04d}.npz")

    def pending(kind, bs):
        return [(b, s) for b in bs for s in range(len(inp.by[b]["plan"][kind]))
                if not os.path.exists(shard_path(b, kind, s))]

    def progress():
        return {b: {k: f"{sum(os.path.exists(shard_path(b, k, s)) for s in range(len(v['plan'][k])))}"
                       f"/{len(v['plan'][k])}" for k in ("emb", "llm")} for b, v in inp.by.items()}

    manifest["progress"] = progress()
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    write_json(mpath, manifest)

    # tokens per second: measured in this process once a stage runs; until then
    # --rates, else the previous process's (manifest)
    rate = dict(zip(("emb", "llm"), args.rates)) if args.rates else \
        dict(manifest.get("rates_tok_per_s") or {"emb": None, "llm": None})
    spent = {"emb": [0.0, 0], "llm": [0.0, 0]}
    todo = {k: sum(inp.tokens(b, k, s) for b, s in pending(k, benchmarks)) for k in rate}
    log(f"to do: {todo['emb']:,} embedding tokens, {todo['llm']:,} rating tokens "
        f"({sum(len(pending(k, benchmarks)) for k in rate)} shards)")

    def eta():
        if any(todo[k] and not rate[k] for k in rate):
            return None
        return sum(todo[k] / rate[k] for k in rate if todo[k])

    for kind, bs in stages:
        work = pending(kind, bs)
        if not work:
            log(f"{kind} {bs}: done already")
            continue
        t_load = time.time()
        if kind == "emb":
            runner = Embedder(emb_path, dtype[args.emb_dtype], args.device,
                              emb_tok.pad_token_id, args.emb_budget, args.max_batch)
        else:
            runner = Rater(llm_path, dtype[args.llm_dtype], args.device, llm_tok.pad_token_id,
                           inp.prompt.digits, args.llm_budget, args.max_batch)
        log(f"{kind} model loaded in {time.time() - t_load:.0f}s; {len(work)} shards of {bs}")
        for n_done, (b, s) in enumerate(work, 1):
            v = inp.by[b]
            shard = v["plan"][kind][s]
            keys = np.array([v["rows"][i]["key"] for i in shard])
            t = time.time()
            if kind == "emb":
                emb = runner.shard([v["emb"][i] for i in shard])
                arrays = {"key": keys, "emb": emb,
                          "tokens": np.array([len(v["emb"][i]) for i in shard], np.int32),
                          "text_tokens": np.array([v["rows"][i]["emb_text_tokens"]
                                                   for i in shard], np.int32),
                          "truncated": np.array([v["rows"][i]["emb_truncated"] for i in shard])}
            else:
                res = runner.shard([v["llm"][i] for i in shard])
                res["task_tokens"] = np.array([v["rows"][i]["task_tokens"] for i in shard])
                res["truncated"] = np.array([v["rows"][i]["task_truncated"] for i in shard])
                arrays = {"key": keys, "probs": res.pop("probs"),
                          **{c: res[c].astype(np.float32) for c in F.LLM_SCALARS}}
            atomic_write(shard_path(b, kind, s), lambda fh: np.savez(fh, **arrays))
            dt = time.time() - t
            tok = inp.tokens(b, kind, s)
            spent[kind][0] += dt
            spent[kind][1] += tok
            if spent[kind][0] > 60 or not rate.get(kind):
                rate[kind] = spent[kind][1] / spent[kind][0]
            todo[kind] -= tok
            left = eta()
            finish = (datetime.now() + timedelta(seconds=left)).strftime("%a %H:%M") \
                if left is not None else "?"
            log(f"{kind} {b} shard {s + 1}/{len(v['plan'][kind])}: {len(shard)} items, "
                f"{tok:,} tok in {dt:.1f}s ({tok / dt:,.0f} tok/s) | stage {n_done}/{len(work)}"
                f" | left {_dur(left)}, finish ~{finish}")
            if n_done % 10 == 0 or n_done == len(work):
                manifest["progress"] = progress()
                manifest["rates_tok_per_s"] = rate
                manifest["updated"] = datetime.now().isoformat(timespec="seconds")
                write_json(mpath, manifest)
        del runner
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    manifest["progress"] = progress()
    manifest["complete"] = not any(pending(k, benchmarks) for k in rate)
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    write_json(mpath, manifest)
    log(f"finished in {_dur(time.time() - t0)}; complete={manifest['complete']}")

    if args.validate:
        report(args.out, benchmarks, spent, full)


def report(out, benchmarks, spent, full):
    """Sanity checks and a runtime estimate after --validate. The Spearman
    correlations against a Rasch difficulty fitted on every pair of the benchmark
    are a smoke signal on a dozen items each, not a result."""
    from scipy.stats import spearmanr
    from paiec import testlike as T
    print("\n=== validation ===")
    allr = []
    for b in benchmarks:
        index, emb, probs, sc = F.load(out, b)
        ok = index.has_emb.to_numpy() & index.has_llm.to_numpy()
        norms = np.linalg.norm(emb[ok].astype(np.float32), axis=1)
        bmap = T.rasch(D.load_pairs([b]))
        diff = np.array([np.nanmean([bmap.get(i, np.nan) for i in ids]) if any(
            i in bmap for i in ids) else np.nan for ids in index.item_ids])[ok]
        s = sc[ok]
        m = np.isfinite(diff)
        r_rat = spearmanr(s.rating[m], diff[m])[0] if m.sum() > 2 else np.nan
        r_nll = spearmanr(s.nll[m], diff[m])[0] if m.sum() > 2 else np.nan
        allr.append(s)
        print(f"{b:18s} n={ok.sum():3d} rating {s.rating.mean():.2f}±{s.rating.std():.2f} "
              f"[{s.rating.min():.2f},{s.rating.max():.2f}] mass min {s.digit_mass.min():.3f} "
              f"H {s.entropy.mean():.2f} nll {s.nll.mean():.2f} "
              f"|emb| [{norms.min():.4f},{norms.max():.4f}] "
              f"spearman(rating,b) {r_rat:+.2f} spearman(nll,b) {r_nll:+.2f} (n={m.sum()})")
        print(f"{'':18s} argmax digits {np.bincount(probs[ok].argmax(1), minlength=10).tolist()}")
    s = pd.concat(allr)
    print(f"all: rating sd {s.rating.std():.3f}, distinct argmax "
          f"{len(set(np.round(s.rating, 1)))}, digit mass min {s.digit_mass.min():.4f} "
          f"mean {s.digit_mass.mean():.4f}, share > 0.9: {(s.digit_mass > 0.9).mean():.2f}")
    est = {}
    for k in ("emb", "llm"):
        t, n = spent[k]
        r = n / t if t else float("nan")
        tot = sum(full.tokens(b, k) for b in full.by)
        est[k] = tot / r
        print(f"{k}: {n:,} tok in {t:.0f}s = {r:,.0f} tok/s; full plan "
              f"{tot:,} tok ({', '.join(f'{b} {full.tokens(b, k):,}' for b in full.by)}) "
              f"-> ~{_dur(est[k])}")
    print(f"estimated full run for {list(full.by)}: ~{_dur(sum(est.values()))} "
          f"(plus ~1 min tokenizing and ~1 min loading per stage)")


if __name__ == "__main__":
    main()
