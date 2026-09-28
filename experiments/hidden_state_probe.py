"""Entropy profiles and hidden-state probes: what Qwen3-4B-Instruct-2507's own
reading of a task says about its difficulty, through linear heads carried to an
unseen benchmark.

The 4B judge's answer (a digit) and its attempts carried nothing that transfers
(docs/findings.md, "The 4B judge, closed out" and "Attempting instead of
judging"). This probe reads the same model's internal state instead, in ONE
forward pass per item and with nothing generated:

  prompt   Agent Psychometrics' instructed form (Ge et al., arXiv:2604.00594):
           one user turn, the task text and then the question QUESTION, in the
           model's chat template with the assistant header appended. The task
           text is llmfeat.item_text's: item_content, a newline, item_features;
           item_content is cut by llmfeat.head_tail to llmfeat.TASK_MAX_TOKENS
           with llmfeat.MARKER between head and tail (the rating prompt's rule),
           item_features follow it whole.
  (a) the token-entropy profile of the task span (Krsteski & Meyer,
           arXiv:2608.05797, whose best transferable family it was): at every
           item_content token (the marker's excluded) the entropy, in nats, of
           the model's full next-token distribution that predicts it, and the
           token's surprisal (-log p); the vocabulary is summed in float32
           (online log-sum-exp, as llm_features.Rater). Profile statistics
           (profile_stats): mean, sd, the nine deciles, the slope over relative
           position, the mean absolute step (total variation per token).
  (b) the hidden state at the last prompt token (the newline after the
           assistant header, where the answer would start): after decoder
           layers 9, 18 and 27 (the residual stream) and after the final norm
           (36, what the output projection reads); and the mean over the
           item_content span after layer 18 and after the final norm.

Items (--stage sample). The four multi-subject parents, unique items as
llm_features builds them (data/features/<b>/index.parquet), with an honest
difficulty (experiments/harness.py oracle_maps: Rasch b without each of the
five subject folds, averaged over the folds, higher = harder). matharena keeps
its text-bearing items (llm4b_close.text_bearing: no image placeholder, at
least 70 characters of content). Up to N_SAMPLE per parent, stratified: quotas
per item_features group (competition, lang, website, paper) proportional to
its size (largest remainder), and within a group a systematic sample over the
items sorted by honest difficulty (a random offset, SEED). real_webagents (233)
and researchcodebench (212) are taken whole. The sample and the targets are
written to data/features/probe/ (gitignored).

--stage extract. The streamed 4B of experiments/llm_features.py (StreamedLM:
one decoder layer at a time over a shard, fp16, exactly the whole model's
arithmetic), shards planned by llmfeat.plan on the sampled items (by length,
then key), each written atomically to data/features/probe/<b>/NNNN.npz;
--resume skips the shards on disk after checking the manifest's config. At
most one language-model process may run on this machine: the stage refuses to
start while another is alive (OTHER_LLM, pgrep), and caps the MPS pool
(PYTORCH_MPS_HIGH_WATERMARK_RATIO).

--stage heads (no language model). Linear, low-dimensional heads fitted to the
honest difficulty standardised within benchmark, leave one benchmark out over
the four parents, every item weighted 1 / its benchmark's count, features
standardised within benchmark (each benchmark's items are visible at run
time; no label is used). Hyperparameters are nested: for held-out parent q,
each configuration is fitted on two of the other three and scored on the
third (Pearson within benchmark), the mean over the three inner folds picks
it, and it is refitted on all three and scored on q. Heads (PRIMARY declared
before any result was seen):
  entropy   ridge on the 13 entropy-profile statistics (primary)
  hidden    per-dimension standardised last-token state, PCA to k <= 32 on the
            training parents, ridge; the layer is chosen by the nested loop
            too (primary)
  secondary: surprisal (the 7 surprisal statistics), profile (both
            profiles), hidden_mean (span-mean states, layer nested), and the
            hidden head at each layer fixed.
Reported per held-out parent: Pearson and Spearman over the benchmark, within
item_features group and within group net of log length (llm4b_close.corr_block,
95% intervals from a bootstrap over groups and over items), the sign, a
random-effects mean over the four parents (DerSimonian-Laird, Fisher z) with its
prediction interval for a new benchmark, and the heads' within-benchmark 5-fold
fit (not transferable; how much the features carry inside a benchmark).

--stage harness. Each head's out-of-fold predictions (x on parent q from the
head fitted without q; HARNESS_HEADS, every head) through experiments/harness.py
on its stored rows of the shipped hier, as llm4b_close's harness stage: harness.eval_covariate (x
standardised within benchmark; the B0 term reads raw x) and score_covariate
with the nested transferred, per-pair, hybrid and B0 lines and the forced ones;
placebo: x permuted within benchmark (N_PLACEBO draws). --stage reference: the
honest difficulty degraded to r on exactly the sampled items (0 elsewhere),
what the harness gives at this coverage.

--stage verdict. KEEP only if a primary head has (i) mean LOBO within-benchmark
Pearson r >= KEEP_R with a positive sign on at least KEEP_SIGNS of the 4
parents, and (ii) a test-like ALC difference <= KEEP_ALC on a nested harness
line (transferred or per-pair); otherwise a replicated null. The stricter
reading (r >= KEEP_R on each of 3 parents) is reported beside it.

Run:  python experiments/hidden_state_probe.py --stage sample      # seconds, CPU
      nohup python experiments/hidden_state_probe.py --stage extract --resume > LOG 2>&1 &
                              # 36 min for 608,366 prompt tokens (283 tokens/s), the only LLM job
      python experiments/hidden_state_probe.py --stage heads       # 3 min, CPU
      python experiments/hidden_state_probe.py --stage harness     # 15 min, one process, 0.5 GB
      python experiments/hidden_state_probe.py --stage reference   # 5 min; may run beside harness
      python experiments/hidden_state_probe.py --stage verdict
      python experiments/hidden_state_probe.py --stage show
Needs data/<benchmark>/, data/features/<benchmark>/index.parquet (llm_features.py),
the harness rows (python experiments/harness.py --stage collect) and, for the
extraction only, Qwen3-4B-Instruct-2507 in the local Hugging Face cache (no
download).
"""
import os
import sys

_EXTRACT = "extract" in sys.argv
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4" if _EXTRACT else "1")
if _EXTRACT:
    # the MPS pool: raise instead of pushing the machine into swap, and give
    # cached blocks back early (fragmentation killed the rating extraction)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import harness as H  # noqa: E402
from experiments import itemcov_eval as ICE  # noqa: E402
from experiments import llm4b_close as L4  # noqa: E402
from paiec import itemcov as IC  # noqa: E402
from paiec import llmfeat as F  # noqa: E402

OUT = os.path.join(ROOT, "results", "hidden_state_probe.json")
FEAT_DIR = os.path.join(ROOT, "data", "features")
PROBE_DIR = os.path.join(FEAT_DIR, "probe")
PARENTS = H.PARENTS
GROUP_KEY = ICE.GROUP_KEY
N_SAMPLE = 400
SEED = 20260927
QUESTION = "How difficult is the above task for an AI agent?"
LAYERS_LAST = (9, 18, 27, 36)       # 36: after the final norm
LAYERS_MEAN = (18, 36)
SHARD_ITEMS = 128
SHARD_TOKENS = 32768
BUDGET = 4096                       # padded tokens per batch
MAX_BATCH = 32
POS_CHUNK = 2048                    # positions per vocabulary pass
VOCAB_CHUNK = 16384
OTHER_LLM = "probe_attempts|llm_|lora|icl_probe|probe_compare|hidden_state"

ENT_STATS = ("mean", "sd") + tuple(f"q{d}" for d in range(10, 100, 10)) + ("slope", "tv")
SUR_STATS = ("mean", "sd", "q10", "q50", "q90", "slope", "tv")
PRIMARY = ("entropy", "hidden")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)  # ridge penalty per mean eigenvalue of X'WX
PCA_K = (4, 8, 16, 32)
H_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
BOOT = 2000
N_PLACEBO = 3
PLACEBO_BOOTS = 200
REF_R = (0.2, 0.3, 0.5)
N_REF = 4
KEEP_R = 0.2
KEEP_SIGNS = 3
KEEP_ALC = -0.001
NESTED = ("transferred nested", "per-pair nested", "hybrid nested", "b0 nested")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _dur(s):
    if s is None or not np.isfinite(s):
        return "?"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def atomic_write(path, write):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, lambda fh: fh.write(json.dumps(H._jsonable(obj), indent=1, sort_keys=True).encode()))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --- the sample -------------------------------------------------------------------------

def stratified(rows, n, seed):
    """rows: (key, group, target). All of them if they fit in n; else quotas per
    group proportional to its size (largest remainder, ties by group name) and,
    within a group, a systematic sample over its items sorted by (target, key)
    from a random offset."""
    if len(rows) <= n:
        return sorted(r[0] for r in rows)
    rng = np.random.default_rng(seed)
    by = {}
    for k, g, t in rows:
        by.setdefault(g, []).append((t, k))
    groups = sorted(by)
    exact = {g: n * len(by[g]) / len(rows) for g in groups}
    quota = {g: int(math.floor(exact[g])) for g in groups}
    rest = n - sum(quota.values())
    for g in sorted(groups, key=lambda g: (-(exact[g] - quota[g]), g))[:rest]:
        quota[g] += 1
    out = []
    for g in groups:
        mem = sorted(by[g])
        k = quota[g]
        if not k:
            continue
        u = rng.random()
        out += [mem[min(int((j + u) * len(mem) / k), len(mem) - 1)][1] for j in range(k)]
    return sorted(set(out))


def stage_sample(args):
    import pandas as pd
    t0 = time.time()
    target, honest, tinfo = L4.honest_targets()
    log(f"sample: honest targets in {time.time() - t0:.0f}s")
    keep_keys = set().union(*(set(v) for v in target.values()))
    tpath = os.path.join(PROBE_DIR, "targets.json")
    write_json(tpath, {"mean": target, "folds": [{k: v for k, v in m.items() if k in keep_keys} for m in honest],
                       "info": tinfo, "provenance": H.provenance()})
    sample = {}
    info = {}
    for b in PARENTS:
        index = pd.read_parquet(os.path.join(FEAT_DIR, b, "index.parquet"))
        items = ICE.load_items(b)
        tgt = target[b]
        rows, n_text, n_diff = [], 0, 0
        for key, ids in zip(index["key"], index["item_ids"]):
            ids = [str(i) for i in ids]
            have = [i for i in ids if i in tgt]
            if not have:
                continue
            n_diff += 1
            item = items[ids[0]]
            if b == "matharena" and not L4.text_bearing({0: item})[0]:
                continue
            n_text += 1
            g = IC.features(item).get(GROUP_KEY[b], "")
            rows.append((key, g, float(np.mean([tgt[i] for i in have]))))
        chosen = stratified(rows, args.n, SEED + PARENTS.index(b))
        sample[b] = chosen
        tv = {k: t for k, _, t in rows}
        info[b] = {"unique": int(len(index)), "with_difficulty": n_diff, "eligible": n_text,
                   "sampled": len(chosen), "groups": len({g for _, g, _ in rows}),
                   "target_mean_sd_eligible": [round(float(np.mean(list(tv.values()))), 3),
                                               round(float(np.std(list(tv.values()))), 3)],
                   "target_mean_sd_sampled": [round(float(np.mean([tv[k] for k in chosen])), 3),
                                              round(float(np.std([tv[k] for k in chosen])), 3)]}
        log(f"{b}: {info[b]}")
    write_json(os.path.join(PROBE_DIR, "sample.json"), {"sample": sample, "info": info, "n": args.n, "seed": SEED})
    log(f"sample: {time.time() - t0:.0f}s -> {PROBE_DIR}")


def load_sample():
    s = read_json(os.path.join(PROBE_DIR, "sample.json"))
    if s is None:
        raise SystemExit("run --stage sample first")
    return s


# --- extraction ---------------------------------------------------------------------------

class ProbePrompt:
    """The instructed prompt bound to one tokenizer: the chat template rendered
    once around llmfeat.PLACEHOLDER; the prefix, item_content (head_tail), the
    item_features line and the suffix tokenized separately and joined as ids,
    so the content span's positions are known exactly."""

    def __init__(self, tok, max_task_tokens=F.TASK_MAX_TOKENS):
        messages = [{"role": "user", "content": F.PLACEHOLDER + "\n\n" + QUESTION}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text.count(F.PLACEHOLDER) != 1:
            raise ValueError("chat template did not keep the task placeholder exactly once")
        self.rendered = text
        pre, post = text.split(F.PLACEHOLDER)
        self.prefix, self.suffix = F.encode(tok, pre), F.encode(tok, post)
        self.marker = F.encode(tok, F.MARKER)
        self.max_task_tokens = max_task_tokens
        self._tok = tok

    def build(self, item):
        """(ids, span start, span stop, mask over the span, content tokens before
        truncation, truncated). The span is item_content's; the mask drops the
        marker's tokens."""
        content = F.encode(self._tok, F._s(item.get("item_content")))
        feats = F.encode(self._tok, "\n" + F._s(item.get("item_features")))
        cut, truncated = F.head_tail(content, self.max_task_tokens, self.marker)
        mask = np.ones(len(cut), bool)
        if truncated:
            head = (self.max_task_tokens - len(self.marker) + 1) // 2
            mask[head:head + len(self.marker)] = False
        start = len(self.prefix)
        return (np.array(self.prefix + cut + feats + self.suffix, np.int32), start, start + len(cut), mask,
                len(content), truncated)

    def describe(self):
        return {"question": QUESTION, "rendered": self.rendered, "marker": F.MARKER,
                "max_task_tokens": self.max_task_tokens, "prefix_tokens": len(self.prefix),
                "suffix_tokens": len(self.suffix), "marker_tokens": len(self.marker),
                "task": "item_content (head_tail, marker) + '\\n' + item_features"}


def other_llm_processes():
    """Other processes matching OTHER_LLM, this one and its ancestors excluded."""
    mine, pid = set(), os.getpid()
    while pid and pid not in mine:
        mine.add(pid)
        try:
            pid = int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True,
                                     text=True).stdout.strip() or 0)
        except ValueError:
            break
    out = subprocess.run(["pgrep", "-f", OTHER_LLM], capture_output=True, text=True).stdout
    pids = [int(t) for t in out.split() if t.isdigit() and int(t) not in mine]
    found = []
    for p in pids:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(p)], capture_output=True,
                             text=True).stdout.strip().replace("\n", " ")
        if cmd and "pgrep" not in cmd:
            found.append(f"{p} {cmd[:200]}")
    return found


class Prober:
    """One pass of the streamed 4B over a shard: the captured hidden states and
    the content span's entropy and surprisal profiles."""

    def __init__(self, path, dev, pad_id):
        import torch
        from experiments import llm_features as LF
        self.torch, self.dev, self.pad_id, self.LF = torch, dev, pad_id, LF
        self.lm = LF.StreamedLM(path, torch.float16, dev)
        self.W = self.lm.head
        self.V = self.W.shape[0]
        self.Hd = self.lm.cfg.hidden_size
        if max(LAYERS_LAST + LAYERS_MEAN) != self.lm.n_layers:
            raise ValueError(f"the model has {self.lm.n_layers} layers, not {max(LAYERS_LAST)}")

    def _ent_sur(self, h, tgts):
        """Entropy (nats) of softmax(h W^T) and the surprisal of tgts, float32,
        the vocabulary in chunks with an online log-sum-exp."""
        torch = self.torch
        n = h.shape[0]
        m = torch.full((n,), -float("inf"), device=self.dev)
        s = torch.zeros(n, device=self.dev)
        t = torch.zeros(n, device=self.dev)
        for v0 in range(0, self.V, VOCAB_CHUNK):
            lg = h @ self.W[v0:v0 + VOCAB_CHUNK].float().T
            nm = torch.maximum(m, lg.amax(1))
            sc = torch.exp(m - nm)
            e = torch.exp(lg - nm[:, None])
            s = s * sc + e.sum(1)
            t = t * sc + (e * lg).sum(1)
            m = nm
            del lg, e
        lse = m + torch.log(s)
        ent = lse - t / s
        tl = (h * self.W[tgts].float()).sum(1)
        return ent, lse - tl

    def shard(self, items):
        """items: (ids, start, stop, mask) per prompt. Returns the arrays of one
        shard file (without keys)."""
        torch, dev = self.torch, self.dev
        n = len(items)
        groups = F.batches([len(x[0]) for x in items], BUDGET, MAX_BATCH)
        padded = [self.LF._pad([items[i][0] for i in b], self.pad_id)[0] for b in groups]
        last = {L: np.zeros((n, self.Hd), np.float16) for L in LAYERS_LAST}
        mean = {L: np.zeros((n, self.Hd), np.float16) for L in LAYERS_MEAN}
        ent = [None] * n
        sur = [None] * n
        lm = self.lm
        with torch.inference_mode():
            lastpos = [torch.tensor([len(items[i][0]) - 1 for i in b], device=dev) for b in groups]
            wspan = []
            for b, ids in zip(groups, padded):
                w = torch.zeros(ids.shape, dtype=torch.float32)
                for r, i in enumerate(b):
                    _, start, stop, keep = items[i]
                    pos = np.arange(start, stop)[keep]
                    w[r, pos] = 1.0 / max(len(pos), 1)
                wspan.append(w.to(dev))
            ar = [torch.arange(len(b), device=dev) for b in groups]

            def capture(j, b, h, L):
                if L in LAYERS_LAST:
                    last[L][b] = h[ar[j], lastpos[j]].float().cpu().numpy().astype(np.float16)
                if L in LAYERS_MEAN:
                    mean[L][b] = torch.bmm(wspan[j][:, None, :], h.float())[:, 0].cpu().numpy() \
                        .astype(np.float16)

            hs = [torch.nn.functional.embedding(ids.to(dev), lm.embed) for ids in padded]
            pos_ids = [torch.arange(h.shape[1], device=dev)[None] for h in hs]
            pe = [lm.rotary(h, p) for h, p in zip(hs, pos_ids)]
            for li in range(lm.n_layers):
                lm._load_layer(li)
                for j in range(len(hs)):
                    out = lm.layer(hs[j], attention_mask=None, position_ids=pos_ids[j],
                                   position_embeddings=pe[j], use_cache=False)
                    hs[j] = out[0] if isinstance(out, tuple) else out
                    if li + 1 < lm.n_layers:
                        capture(j, groups[j], hs[j], li + 1)
                torch.mps.empty_cache() if dev == "mps" else None
            for j, b in enumerate(groups):
                h = lm.norm(hs[j])
                hs[j] = None
                if not torch.isfinite(h[ar[j], lastpos[j]]).all():
                    raise FloatingPointError("non-finite hidden state")
                capture(j, b, h, lm.n_layers)
                L = h.shape[1]
                flat = h.reshape(-1, h.shape[-1])
                rows, tgts, owner = [], [], []
                for r, i in enumerate(b):
                    seq, start, stop, keep = items[i]
                    pos = np.arange(start, stop)[keep]
                    rows.append(r * L + pos - 1)
                    tgts.append(seq[pos])
                    owner.append(np.full(len(pos), r))
                rows = torch.from_numpy(np.concatenate(rows)).to(dev)
                tgts = torch.from_numpy(np.concatenate(tgts).astype(np.int64)).to(dev)
                owner = np.concatenate(owner)
                es, ss = [], []
                for p0 in range(0, len(rows), POS_CHUNK):
                    e, s_ = self._ent_sur(flat[rows[p0:p0 + POS_CHUNK]].float(), tgts[p0:p0 + POS_CHUNK])
                    es.append(e.cpu().numpy())
                    ss.append(s_.cpu().numpy())
                es, ss = np.concatenate(es), np.concatenate(ss)
                if not (np.isfinite(es).all() and np.isfinite(ss).all()):
                    raise FloatingPointError("non-finite entropy or surprisal")
                for r, i in enumerate(b):
                    ent[i] = es[owner == r].astype(np.float32)
                    sur[i] = ss[owner == r].astype(np.float32)
                del h, flat
            if dev == "mps":
                torch.mps.empty_cache()
        off = np.concatenate([[0], np.cumsum([len(e) for e in ent])]).astype(np.int64)
        return {"ent": np.concatenate(ent), "sur": np.concatenate(sur), "off": off,
                **{f"last_{L}": v for L, v in last.items()}, **{f"mean_{L}": v for L, v in mean.items()}}


def probe_inputs(tok, sample):
    """{bench: (keys, [(ids, start, stop, mask)], meta rows, plan)} for the sample."""
    import pandas as pd
    prompt = ProbePrompt(tok)
    out = {}
    for b in PARENTS:
        index = pd.read_parquet(os.path.join(FEAT_DIR, b, "index.parquet"))
        first = {k: str(ids[0]) for k, ids in zip(index["key"], index["item_ids"])}
        items = ICE.load_items(b)
        keys = list(sample[b])
        inp, meta = [], []
        for k in keys:
            ids, start, stop, mask, n_content, trunc = prompt.build(items[first[k]])
            inp.append((ids, start, stop, mask))
            meta.append({"prompt_tokens": len(ids), "content_tokens": n_content, "truncated": trunc,
                         "span": int(mask.sum())})
        plan = F.plan([len(x[0]) for x in inp], keys, SHARD_ITEMS, SHARD_TOKENS)
        out[b] = (keys, inp, meta, plan)
    return prompt, out


def stage_extract(args):
    others = other_llm_processes()
    if others and not args.force:
        raise SystemExit("another language-model process is alive; at most one may run:\n" + "\n".join(others))
    import torch
    import transformers
    from transformers import AutoTokenizer
    from experiments import llm_features as LF
    torch.manual_seed(0)
    path, rev = LF.snapshot(F.LLM_REPO)
    tok = AutoTokenizer.from_pretrained(path)
    sample = load_sample()
    t0 = time.time()
    prompt, inp = probe_inputs(tok, sample["sample"])
    log(f"tokenized the sample in {time.time() - t0:.0f}s")
    config = {"model": F.LLM_REPO, "revision": rev, "prompt": prompt.describe(), "layers_last": LAYERS_LAST,
              "layers_mean": LAYERS_MEAN, "dtype": "float16", "device": args.device,
              "readout": "entropy of the full next-token softmax (float32, online log-sum-exp) and surprisal "
                         "of each item_content token (marker excluded) from the hidden state before it; "
                         "last-token states after layers 9, 18, 27 (residual stream) and after the final "
                         "norm (36); span means after layer 18 and the final norm",
              "shard_items": SHARD_ITEMS, "shard_tokens": SHARD_TOKENS, "budget": BUDGET, "max_batch": MAX_BATCH,
              "runner": "experiments/llm_features.StreamedLM",
              "sample_digest": F.digest(sample["sample"]),
              "plan_digest": F.digest({b: [[v[0][i] for i in s] for s in v[3]] for b, v in inp.items()})}
    chash = F.digest(config)
    mpath = os.path.join(PROBE_DIR, "manifest.json")
    man = read_json(mpath)
    if man is not None:
        if not args.resume:
            raise SystemExit(f"{mpath} exists: pass --resume")
        if man.get("config_hash") != chash:
            raise SystemExit("the manifest was written for another prompt, sample or plan; refusing to mix")
    else:
        man = {"config": config, "config_hash": chash, "started": datetime.now().isoformat(timespec="seconds"),
               "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                       "numpy": np.__version__, "argv": sys.argv,
                       "mps_watermarks": [os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                                          os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")]}}

    def spath(b, s):
        return os.path.join(PROBE_DIR, b, f"{s:04d}.npz")

    order = [b for b in args.benchmarks if b in inp]
    work = [(b, s) for b in order for s in range(len(inp[b][3])) if not os.path.exists(spath(b, s))]
    tok_of = {(b, s): int(sum(len(inp[b][1][i][0]) for i in inp[b][3][s])) for b in inp for s in range(len(inp[b][3]))}
    man["benchmarks"] = {b: {"items": len(v[0]), "shards": len(v[3]),
                             "tokens": int(sum(len(x[0]) for x in v[1])),
                             "truncated": int(sum(m["truncated"] for m in v[2]))} for b, v in inp.items()}
    write_json(mpath, man)
    todo = sum(tok_of[w] for w in work)
    log(f"to do: {len(work)} shards, {todo:,} tokens; " + ", ".join(
        f"{b} {man['benchmarks'][b]}" for b in order))
    if not work:
        return
    t_load = time.time()
    prober = Prober(path, args.device, tok.pad_token_id)
    log(f"model loaded in {time.time() - t_load:.0f}s")
    spent, done = 0.0, 0
    for n_done, (b, s) in enumerate(work, 1):
        keys, items, meta, plan = inp[b]
        shard = plan[s]
        t = time.time()
        arr = prober.shard([items[i] for i in shard])
        arr.update({"key": np.array([keys[i] for i in shard]),
                    **{c: np.array([meta[i][c] for i in shard]) for c in ("prompt_tokens", "content_tokens",
                                                                           "truncated", "span")}})
        atomic_write(spath(b, s), lambda fh: np.savez(fh, **arr))
        dt = time.time() - t
        spent += dt
        done += tok_of[(b, s)]
        rate = done / spent
        left = (todo - done) / rate
        mem = torch.mps.driver_allocated_memory() / 2 ** 30 if args.device == "mps" else float("nan")
        log(f"{b} shard {s + 1}/{len(plan)}: {len(shard)} items, {tok_of[(b, s)]:,} tok in {dt:.0f}s "
            f"({tok_of[(b, s)] / dt:,.0f} tok/s), MPS driver {mem:.1f} GB | {n_done}/{len(work)} | left "
            f"{_dur(left)}, finish ~{(datetime.now() + timedelta(seconds=left)).strftime('%H:%M')}")
        man["progress"] = {bb: f"{sum(os.path.exists(spath(bb, ss)) for ss in range(len(inp[bb][3])))}/"
                               f"{len(inp[bb][3])}" for bb in inp}
        man["rate_tok_per_s"] = rate
        man["updated"] = datetime.now().isoformat(timespec="seconds")
        write_json(mpath, man)
    man["complete"] = all(os.path.exists(spath(b, s)) for b in inp for s in range(len(inp[b][3])))
    write_json(mpath, man)
    log(f"extract: {_dur(time.time() - t0)}; complete={man['complete']}")


# --- the probe's data ------------------------------------------------------------------------

def profile_stats(v, which):
    """Statistics of one per-token profile (ENT_STATS or SUR_STATS): mean, sd,
    deciles, the OLS slope over relative position in [0, 1], and the mean
    absolute step between neighbouring tokens."""
    v = np.asarray(v, np.float64)
    n = len(v)
    if n == 0:
        return np.full(len(which), np.nan)
    t = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    d = {"mean": v.mean(), "sd": v.std(),
         "slope": float(np.polyfit(t, v, 1)[0]) if n > 2 else 0.0,
         "tv": float(np.abs(np.diff(v)).mean()) if n > 1 else 0.0}
    for q in range(10, 100, 10):
        d[f"q{q}"] = float(np.percentile(v, q))
    return np.array([d[s] for s in which], float)


class Probe:
    """The extracted features at item_id level (every item_id a sampled unique
    item stands for), with the honest target, group, log length and position.
    Attributes per row: bench, iid, key, y (honest difficulty), g, L (log
    length), pos (matharena's position or NaN); feature blocks in `blocks`."""

    def __init__(self, benches=PARENTS, need_complete=False):
        import pandas as pd
        sample = load_sample()["sample"]
        tg = read_json(os.path.join(PROBE_DIR, "targets.json"))
        self.target, self.folds = tg["mean"], tg["folds"]
        rows = {c: [] for c in ("bench", "iid", "key", "y", "g", "L", "pos", "n_span", "truncated")}
        blocks = {"entropy": [], "surprisal": []}
        blocks.update({f"last_{L}": [] for L in LAYERS_LAST})
        blocks.update({f"mean_{L}": [] for L in LAYERS_MEAN})
        self.coverage = {}
        for b in benches:
            d = os.path.join(PROBE_DIR, b)
            files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and not f.startswith(".")) \
                if os.path.isdir(d) else []
            got = {}
            for f in files:
                with np.load(os.path.join(d, f)) as z:
                    off = z["off"]
                    for r, k in enumerate(z["key"]):
                        got[str(k)] = {"ent": z["ent"][off[r]:off[r + 1]], "sur": z["sur"][off[r]:off[r + 1]],
                                       **{n: z[n][r] for n in blocks if n.startswith(("last_", "mean_"))},
                                       "truncated": bool(z["truncated"][r])}
            self.coverage[b] = {"sampled": len(sample.get(b, [])), "extracted": len(got)}
            if need_complete and len(got) < len(sample.get(b, [])):
                raise SystemExit(f"{b}: {len(got)} of {len(sample[b])} sampled items extracted")
            if not got:
                continue
            index = pd.read_parquet(os.path.join(FEAT_DIR, b, "index.parquet"))
            ids_of = {k: [str(i) for i in ids] for k, ids in zip(index["key"], index["item_ids"])}
            items = ICE.load_items(b)
            tgt = self.target[b]
            for k in sorted(got):
                v = got[k]
                e, s = profile_stats(v["ent"], ENT_STATS), profile_stats(v["sur"], SUR_STATS)
                for iid in ids_of[k]:
                    if iid not in tgt:
                        continue
                    it = items[iid]
                    p = IC.position(it)
                    for c, x in (("bench", b), ("iid", iid), ("key", k), ("y", tgt[iid]),
                                 ("g", IC.features(it).get(GROUP_KEY[b], "")), ("L", IC.log_length(it)),
                                 ("pos", np.nan if p is None else p), ("n_span", len(v["ent"])),
                                 ("truncated", v["truncated"])):
                        rows[c].append(x)
                    blocks["entropy"].append(e)
                    blocks["surprisal"].append(s)
                    for n in blocks:
                        if n.startswith(("last_", "mean_")):
                            blocks[n].append(v[n])
        for c, v in rows.items():
            setattr(self, c, np.array(v, object if c in ("bench", "iid", "key", "g") else float))
        self.blocks = {n: np.array(v, np.float32 if n.startswith(("last_", "mean_")) else np.float64)
                       for n, v in blocks.items()}
        self.benches = [b for b in benches if (self.bench == b).any()]
        self.n = len(self.y)


def zscore_within(X, bench):
    """Columns standardised within each benchmark (0 where a column is constant)."""
    X = np.asarray(X, np.float64)
    out = np.zeros_like(X)
    for b in np.unique(bench):
        m = bench == b
        mu, sd = X[m].mean(0), X[m].std(0)
        out[m] = np.where(sd > 0, (X[m] - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    return out


def weights(bench):
    """1 / the benchmark's count: every benchmark weighs the same."""
    u, inv, cnt = np.unique(bench, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def ridge(X, y, w, alpha):
    """Weighted ridge without intercept (X and y are centred within benchmark);
    alpha relative to the mean eigenvalue of X'WX."""
    A = (X * w[:, None]).T @ X
    lam = alpha * np.trace(A) / max(A.shape[0], 1)
    return np.linalg.solve(A + lam * np.eye(A.shape[0]), (X * w[:, None]).T @ y)


def pear(a, b):
    a, b = np.asarray(a, float) - np.mean(a), np.asarray(b, float) - np.mean(b)
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else 0.0


# --- heads -----------------------------------------------------------------------------------

class ProfileHead:
    """Ridge on standardised profile statistics; configs: alpha."""

    def __init__(self, P, cols):
        self.X = zscore_within(np.column_stack([P.blocks[c] for c in cols]), P.bench)
        self.configs = [(a,) for a in ALPHAS]

    def fit_predict(self, P, train, test, cfg, yz):
        w = weights(P.bench[train])
        beta = ridge(self.X[train], yz[train], w, cfg[0])
        return self.X[test] @ beta, {"beta": beta}


class HiddenHead:
    """Per-dimension standardised states (within benchmark), PCA to k on the
    training rows, ridge on the k scores; configs: (block, k, alpha)."""

    def __init__(self, P, blocks):
        self.Z = {b: zscore_within(P.blocks[b], P.bench).astype(np.float32) for b in blocks}
        self.configs = [(b, k, a) for b in blocks for k in PCA_K for a in H_ALPHAS]
        self._svd = {}

    def _pcs(self, block, train):
        key = (block, train.tobytes())
        if key not in self._svd:
            Xt = self.Z[block][train]
            mu = Xt.mean(0)
            _, s, vt = np.linalg.svd(Xt - mu, full_matrices=False)
            self._svd[key] = (mu, vt[:max(PCA_K)].copy(), s)
        return self._svd[key]

    def fit_predict(self, P, train, test, cfg, yz):
        block, k, a = cfg
        mu, vt, s = self._pcs(block, train)
        V = vt[:k].T
        St, Sq = (self.Z[block][train] - mu) @ V, (self.Z[block][test] - mu) @ V
        w = weights(P.bench[train])
        beta = ridge(St, yz[train], w, a)
        return Sq @ beta, {"explained": float((s[:k] ** 2).sum() / (s ** 2).sum())}


def head_specs(P):
    """name -> (factory, primary?). The factories build on P."""
    return {
        "entropy": (lambda: ProfileHead(P, ["entropy"]), True),
        "hidden": (lambda: HiddenHead(P, [f"last_{L}" for L in LAYERS_LAST]), True),
        "surprisal": (lambda: ProfileHead(P, ["surprisal"]), False),
        "profile": (lambda: ProfileHead(P, ["entropy", "surprisal"]), False),
        "hidden_mean": (lambda: HiddenHead(P, [f"mean_{L}" for L in LAYERS_MEAN]), False),
        **{f"hidden_L{L}": ((lambda L=L: HiddenHead(P, [f"last_{L}"])), False) for L in LAYERS_LAST},
    }


def lobo(P, head, yz):
    """Nested leave-one-benchmark-out: OOF predictions, the chosen config and the
    inner criteria per held-out benchmark. Configs are scored block by block so
    the PCA of one training set is computed once."""
    oof = np.full(P.n, np.nan)
    folds = {}
    for q in P.benches:
        test = P.bench == q
        train_b = [b for b in P.benches if b != q]
        crit = {}
        for c in head.configs:
            rs = []
            for q2 in train_b:
                tr = np.isin(P.bench, [b for b in train_b if b != q2])
                te = P.bench == q2
                pred, _ = head.fit_predict(P, tr, te, c, yz)
                rs.append(pear(pred, P.y[te]))
            crit[c] = (float(np.mean(rs)), rs)
        best = max(head.configs, key=lambda c: (crit[c][0], -head.configs.index(c)))
        tr = np.isin(P.bench, train_b)
        pred, extra = head.fit_predict(P, tr, test, best, yz)
        oof[test] = pred
        folds[q] = {"choice": list(best), "inner_mean_r": crit[best][0], "inner_r": crit[best][1],
                    "inner_best_by_first": _best_by_first(head, crit),
                    **({"explained": extra["explained"]} if "explained" in extra else {}),
                    **({"beta": [round(float(v), 4) for v in extra["beta"]]} if "beta" in extra else {})}
    return oof, folds


def _best_by_first(head, crit):
    """For configs with a first element that names a block: the best inner mean
    per block (how the layers compare on the inner folds)."""
    out = {}
    for c, (m, _) in crit.items():
        if isinstance(c[0], str):
            out[c[0]] = max(out.get(c[0], -9.0), round(m, 4))
    return out or None


def within_cv(P, head, cfg, yz, seed=0):
    """Within-benchmark 5-fold (items at random) out-of-fold r with a fixed
    config: what the features carry inside a benchmark, not transferable."""
    out = {}
    for q in P.benches:
        idx = np.flatnonzero(P.bench == q)
        # folds over unique items, so a duplicate text never straddles folds
        ukeys = np.unique(P.key[idx])
        fk = {k: i % CV_FOLDS for i, k in enumerate(np.random.default_rng([seed, len(idx)]).permutation(ukeys))}
        f = np.array([fk[k] for k in P.key[idx]])
        pred = np.zeros(len(idx))
        for j in range(CV_FOLDS):
            tr = np.zeros(P.n, bool)
            tr[idx[f != j]] = True
            te = np.zeros(P.n, bool)
            te[idx[f == j]] = True
            pred[f == j], _ = head.fit_predict(P, tr, te, cfg, yz)
        g = P.g[idx]
        out[q] = {"pearson": round(pear(pred, P.y[idx]), 4),
                  "pearson_within_group": round(pear(_demean(pred, g), _demean(P.y[idx], g)), 4)}
    return out


def _demean(v, g):
    v = np.asarray(v, float).copy()
    for u in np.unique(g):
        m = g == u
        v[m] -= v[m].mean()
    return v


def random_effects(rs, ns):
    """DerSimonian-Laird on Fisher z: mean r, its 95% CI, tau, and the 95%
    prediction interval for a new benchmark (t with k - 2 df)."""
    from scipy import stats as SS
    rs, ns = np.asarray(rs, float), np.asarray(ns, float)
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    v = 1.0 / np.maximum(ns - 3, 1)
    w = 1 / v
    zf = (w * z).sum() / w.sum()
    Q = (w * (z - zf) ** 2).sum()
    k = len(z)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    zr = (ws * z).sum() / ws.sum()
    se = math.sqrt(1 / ws.sum())
    t = SS.t.ppf(0.975, k - 2) if k > 2 else float("nan")
    pi = math.sqrt(tau2 + se ** 2) * t
    return {"mean": round(float(np.tanh(zr)), 4), "ci": [round(float(np.tanh(zr - 1.96 * se)), 4),
                                                          round(float(np.tanh(zr + 1.96 * se)), 4)],
            "tau_z": round(math.sqrt(tau2), 4), "Q": round(float(Q), 3),
            "prediction_interval": [round(float(np.tanh(zr - pi)), 4), round(float(np.tanh(zr + pi)), 4)]}


def controls(P, m):
    """llm4b_close.corr_block's controls: log length, and position where every
    row has one (matharena)."""
    L = P.L[m]
    pos = P.pos[m]
    return np.column_stack([L, pos]) if np.isfinite(pos).all() else L


def single_signs(P, boots):
    """Each profile statistic alone against honest difficulty per parent:
    Spearman over the benchmark and within group (sign table, no head)."""
    out = {}
    for blk, names in (("entropy", ENT_STATS), ("surprisal", SUR_STATS)):
        for j, s in enumerate(names):
            per = {}
            for q in P.benches:
                m = P.bench == q
                x = P.blocks[blk][m, j]
                cb = L4.corr_block(x, P.y[m], P.g[m], controls(P, m), boots, seed=31 * j + len(q))
                per[q] = {k: cb[k]["est"] for k in ("spearman", "spearman_within", "partial_spearman") if k in cb}
                per[q]["ci_within_group"] = cb.get("spearman_within", {}).get("ci_group")
            sw = [per[q]["spearman_within"] for q in per]
            out[f"{blk}.{s}"] = {"units": per, "signs_within": [int(np.sign(v)) for v in sw],
                                 "same_sign_within": int(max(sum(v > 0 for v in sw), sum(v < 0 for v in sw)))}
    return out


def nll_check(P):
    """A validity check of the extraction: the probe's mean surprisal of
    item_content against the rating extraction's nll (llmfeat: the same 4B, the
    rating prompt, the whole task text), over the unique items both hold."""
    out = {}
    for b in P.benches:
        try:
            index, _, _, sc = F.load(FEAT_DIR, b)
        except FileNotFoundError:
            continue
        nll = dict(zip(index["key"], sc["nll"].to_numpy()))
        m = P.bench == b
        ks, first = np.unique(P.key[m], return_index=True)
        mine = P.blocks["surprisal"][m][first, SUR_STATS.index("mean")]
        other = np.array([nll.get(k, np.nan) for k in ks])
        ok = np.isfinite(other)
        if ok.sum() >= 20:
            out[b] = {"n": int(ok.sum()), "pearson": round(pear(mine[ok], other[ok]), 4),
                      "spearman": round(pear(L4.ranks(mine[ok]), L4.ranks(other[ok])), 4)}
    return out


def stage_heads(args):
    t0 = time.time()
    P = Probe(need_complete=not args.partial)
    log(f"heads: {P.n} rows over {P.benches} in {time.time() - t0:.0f}s; coverage {P.coverage}")
    yz = zscore_within(P.y[:, None], P.bench)[:, 0]
    specs = head_specs(P)
    names = args.heads or list(specs)
    state = H.load_json(args.out) or {}
    res = state.get("heads", {}) if not args.redo else {}
    oof_path = os.path.join(PROBE_DIR, "oof.json")
    oof_all = (read_json(oof_path) or {}) if not args.redo else {}
    for name in names:
        t1 = time.time()
        make, primary = specs[name]
        head = make()
        oof, folds = lobo(P, head, yz)
        per = {}
        for i, q in enumerate(P.benches):
            m = P.bench == q
            cb = L4.corr_block(oof[m], P.y[m], P.g[m], controls(P, m), args.boots, seed=97 * i + 11)
            per[q] = cb
        pr = [per[q]["pearson"]["est"] for q in P.benches]
        sp = [per[q]["spearman"]["est"] for q in P.benches]
        pw = [per[q]["pearson_within"]["est"] for q in P.benches]
        ns = [per[q]["n"] for q in P.benches]
        # the within-benchmark fit with the config the outer folds chose most often
        cfg = max((tuple(f["choice"]) for f in folds.values()),
                  key=lambda c: sum(tuple(f["choice"]) == c for f in folds.values()))
        entry = {"primary": primary, "folds": folds, "per_parent": per,
                 "pearson": dict(zip(P.benches, pr)), "spearman": dict(zip(P.benches, sp)),
                 "pearson_within_group": dict(zip(P.benches, pw)),
                 "mean_pearson": round(float(np.mean(pr)), 4), "mean_spearman": round(float(np.mean(sp)), 4),
                 "mean_pearson_within_group": round(float(np.mean(pw)), 4),
                 "positive_parents": int(sum(r > 0 for r in pr)),
                 "parents_r_at_least": int(sum(r >= KEEP_R for r in pr)),
                 "random_effects": random_effects(pr, ns),
                 "random_effects_within_group": random_effects(pw, ns),
                 "within_benchmark_cv": {"config": list(cfg), **within_cv(P, head, cfg, yz)},
                 "wall_s": round(time.time() - t1, 1)}
        entry["r_prong"] = bool(entry["mean_pearson"] >= KEEP_R and entry["positive_parents"] >= KEEP_SIGNS)
        entry["r_prong_strict"] = bool(entry["parents_r_at_least"] >= KEEP_SIGNS
                                       and entry["positive_parents"] >= KEEP_SIGNS)
        res[name] = entry
        oof_all[name] = {iid: float(x) for iid, x in zip(P.iid, oof)}
        log(f"{name:12s} r " + "  ".join(f"{q[:10]} {r:+.3f}/{w:+.3f}" for q, r, w in zip(P.benches, pr, pw))
            + f" | mean {entry['mean_pearson']:+.3f} (within group {entry['mean_pearson_within_group']:+.3f}),"
              f" RE {entry['random_effects']['mean']:+.3f} PI {entry['random_effects']['prediction_interval']}"
              f" | choices {[f['choice'] for f in folds.values()]} | within-bench CV "
            + str({q: v['pearson'] for q, v in entry['within_benchmark_cv'].items() if q != 'config'})
            + f" | {entry['wall_s']}s")
        state["heads"] = res
        H.save_json(args.out, state)
        write_json(oof_path, oof_all)
    t1 = time.time()
    state["single_features"] = single_signs(P, min(args.boots, 500))
    log(f"single features in {time.time() - t1:.0f}s")
    sample = load_sample()
    man = read_json(os.path.join(PROBE_DIR, "manifest.json")) or {}
    state["meta"] = {**H.provenance(), "script_digest": H.digest(["experiments/hidden_state_probe.py"]),
                     "sample": sample["info"], "coverage": P.coverage, "rows": P.n,
                     "nll_check": nll_check(P),
                     "span_tokens": {b: {"median": float(np.median(P.n_span[P.bench == b])),
                                         "truncated_rows": int(P.truncated[P.bench == b].sum())} for b in P.benches},
                     "extraction": {k: man.get(k) for k in ("config_hash", "benchmarks", "rate_tok_per_s",
                                                            "complete", "started", "updated")},
                     "prompt": (man.get("config") or {}).get("prompt"), "layers_last": LAYERS_LAST,
                     "layers_mean": LAYERS_MEAN, "ent_stats": ENT_STATS, "sur_stats": SUR_STATS,
                     "grids": {"alphas": ALPHAS, "pca_k": PCA_K, "hidden_alphas": H_ALPHAS},
                     "primary": PRIMARY, "keep": {"r": KEEP_R, "signs": KEEP_SIGNS, "alc": KEEP_ALC},
                     "target": "honest Rasch difficulty (harness.oracle_maps), mean over 5 subject folds"}
    H.save_json(args.out, state)
    log(f"heads: {time.time() - t0:.0f}s -> {args.out}")


# --- harness -----------------------------------------------------------------------------------

HARNESS_HEADS = ("entropy", "hidden", "surprisal", "profile", "hidden_mean", "hidden_L9", "hidden_L18",
                 "hidden_L27", "hidden_L36")
FORCED_ALL = L4.FORCED_ALL


def oof_digest():
    with open(os.path.join(PROBE_DIR, "oof.json"), "rb") as f:
        return F.digest({"oof": __import__("hashlib").sha256(f.read()).hexdigest()})[:16]


def stage_harness(args):
    """Each head's out-of-fold predictions as a covariate on the stored rows of
    the shipped hier (llm4b_close.stage_harness's recipe): the harness's nested
    and forced lines, the within-pair r, and a placebo (x permuted within
    benchmark)."""
    t0 = time.time()
    oof = read_json(os.path.join(PROBE_DIR, "oof.json"))
    if not oof:
        raise SystemExit("run --stage heads first")
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    items_bench = H.benchmark_items(rows, keys)
    tg = read_json(os.path.join(PROBE_DIR, "targets.json"))
    honest = tg["folds"]
    odig = oof_digest()
    prov = H.provenance()
    log(f"harness: rows, targets and predictions in {time.time() - t0:.0f}s")
    state = H.load_json(args.out) or {}
    res = state.get("harness", {})
    if res.get("_meta", {}).get("oof_digest") != odig or res.get("_meta", {}).get("lib_digest") != prov["lib_digest"]:
        res = {}
    res["_meta"] = {"oof_digest": odig, "lib_digest": prov["lib_digest"], "rows_lib_digest": H.rows_digest(rows),
                    "forced": [H.cname(c) for c in FORCED_ALL], "n_placebo": args.placebo,
                    "placebo_boots": PLACEBO_BOOTS, "boots": H.BOOTS,
                    "input": "out-of-fold head predictions, standardised within benchmark "
                             "(harness.eval_covariate); B0 term on raw x"}
    for name in (args.heads or HARNESS_HEADS):
        if name not in oof:
            log(f"{name}: no predictions")
            continue
        if name in res and not args.redo:
            log(f"{name}: kept")
            continue
        t1 = time.time()
        raw = oof[name]
        cov, info = H.eval_covariate(raw, items_bench, keys, name, standardise=True)
        cov.keys = keys
        covered = {reg: float(np.mean(np.isin(np.array(keys, object)[R.ev_k], list(raw)))) for reg, R in rows.items()}
        lines, _ = H.score_covariate(rows, cov, forced=FORCED_ALL)
        entry = {"x": info, "coverage_eval_items": covered, **ICE.realised_r(rows, cov, honest),
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
        _save_part(args.out, "harness", res)
        a = entry["lines"]
        log(f"{name}: {entry['wall_s']}s  r_within_pair {entry.get('r_within_pair_tl')}  " + "  ".join(
            f"{n.split()[0]} {a[n]['tl']:+.5f} (eq {a[n]['tl_benchmark_equal']:+.5f}, on {a[n]['folds_on']})"
            for n in NESTED) + "  forced: " + "  ".join(f"{n} {a[n]['tl']:+.5f}" for n in a if n.endswith("(forced)")))
    _save_part(args.out, "harness", res)
    log(f"harness: {time.time() - t0:.0f}s -> {args.out}")


def _save_part(path, part, value):
    """Write one part of the results file, re-reading the rest first (the
    harness and reference stages may run side by side)."""
    st = H.load_json(path) or {}
    st[part] = value
    H.save_json(path, st)


def stage_reference(args):
    """The honest difficulty degraded to r on exactly the sampled items (0
    elsewhere), N_REF draws per r: what the harness can give at this coverage."""
    t0 = time.time()
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    tg = read_json(os.path.join(PROBE_DIR, "targets.json"))
    target, honest = tg["mean"], tg["folds"]
    oof = read_json(os.path.join(PROBE_DIR, "oof.json")) or {}
    covered = set(next(iter(oof.values()))) if oof else set()
    par_of_key = {k: par for par, d in target.items() for k in d}
    Z, has, _ = H.base_matrix(honest, keys, par_of_key)
    on = np.array([k in covered and k in par_of_key for k in keys])
    out = {"covered_keys_in_rows": int(on.sum()), "r": {}}
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
        log(f"reference r={r:g}: " + "  ".join(f"{n} {v['tl']:+.5f} (eq {v['tl_benchmark_equal']:+.5f})"
                                               for n, v in out["r"][f"r={r:g}"].items()))
    out["wall_s"] = round(time.time() - t0, 1)
    _save_part(args.out, "reference", out)


# --- verdict and show ---------------------------------------------------------------------------

RULE_LINES = ("transferred nested", "per-pair nested")


def stage_verdict(args):
    """KEEP a primary head only if its r prong (mean LOBO within-benchmark
    Pearson >= KEEP_R, positive on >= KEEP_SIGNS parents) and its ALC prong
    (a nested test-like difference <= KEEP_ALC on RULE_LINES) both hold. The
    secondary heads are read the same way and reported, not decided on."""
    state = H.load_json(args.out)
    if not state or "heads" not in state or "harness" not in state:
        raise SystemExit("run --stage heads and --stage harness first")
    per = {}
    for name, e in state["heads"].items():
        hz = state["harness"].get(name)
        alc = None if not hz else {n: hz["lines"][n]["tl"] for n in hz["lines"]
                                   if n in NESTED or n.endswith("(forced)")}
        alc_prong = None if alc is None else any(alc[n] <= KEEP_ALC for n in RULE_LINES)
        per[name] = {"primary": e["primary"], "pearson": e["pearson"], "mean_pearson": e["mean_pearson"],
                     "positive_parents": e["positive_parents"], "parents_r_at_least": e["parents_r_at_least"],
                     "r_prong": e["r_prong"], "r_prong_strict": e["r_prong_strict"],
                     "alc_tl": alc, "alc_prong": alc_prong,
                     "keep": bool(e["r_prong"] and alc_prong)}
    keep = [n for n, v in per.items() if v["primary"] and v["keep"]]
    exploratory = [n for n, v in per.items() if not v["primary"] and v["keep"]]
    state["verdict"] = {
        "rule": (f"keep a primary head only if its mean leave-one-benchmark-out within-benchmark Pearson r is >= "
                 f"{KEEP_R} with a positive sign on >= {KEEP_SIGNS} of 4 parents AND a nested test-like harness "
                 f"ALC difference (transferred or per-pair) is <= {KEEP_ALC}"),
        "heads": per, "keep": keep, "exploratory_pass": exploratory,
        "script_digest": H.digest(["experiments/hidden_state_probe.py"]),
        "call": ("KEEP " + ", ".join(keep)) if keep else "NULL (replicated): no primary head passes"}
    H.save_json(args.out, state)
    print(json.dumps(state["verdict"], indent=1))


def _ci(v, nd=2):
    if not v or v.get("est") is None:
        return ""
    ci = v.get("ci_group") or v.get("ci_item")
    return f"{v['est']:+.{nd}f}" + ("" if not ci else f" [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]")


def stage_show(args):
    s = H.load_json(args.out)
    m = s.get("meta", {})
    print("sample:", json.dumps(m.get("sample")), "\ncoverage:", m.get("coverage"))
    hd = s.get("heads", {})
    if hd:
        print("\n| head | " + " | ".join(PARENTS) + " | mean | positive | RE mean [PI] |\n|" + "---|" * 8)
        for n, e in hd.items():
            print(f"| {n}{' (primary)' if e['primary'] else ''} | "
                  + " | ".join(_ci(e["per_parent"][q].get("pearson")) for q in PARENTS if q in e["per_parent"])
                  + f" | {e['mean_pearson']:+.3f} | {e['positive_parents']}/4 | {e['random_effects']['mean']:+.3f} "
                    f"{e['random_effects']['prediction_interval']} |")
        for stat in ("spearman", "pearson_within", "spearman_within", "partial_spearman", "partial_pearson"):
            print(f"\n{stat}\n| head | " + " | ".join(PARENTS) + " |\n|" + "---|" * 5)
            for n, e in hd.items():
                print(f"| {n} | " + " | ".join(_ci(e["per_parent"][q].get(stat)) for q in PARENTS
                                               if q in e["per_parent"]) + " |")
        print("\nchoices and within-benchmark CV:")
        for n, e in hd.items():
            print(f"  {n}: " + "; ".join(f"{q[:10]} {f['choice']} inner {f['inner_mean_r']:+.3f}"
                                         + (f" by block {f['inner_best_by_first']}" if f.get("inner_best_by_first")
                                            else "") for q, f in e["folds"].items())
                  + f"\n      CV {e['within_benchmark_cv']}")
    sf = s.get("single_features")
    if sf:
        print("\nsingle statistics, Spearman within group (sign table):\n| stat | " + " | ".join(PARENTS)
              + " | same sign |\n|" + "---|" * 6)
        for n, v in sf.items():
            print(f"| {n} | " + " | ".join(f"{v['units'][q]['spearman_within']:+.3f}" for q in PARENTS
                                           if q in v["units"]) + f" | {v['same_sign_within']}/4 |")
    hz = s.get("harness")
    if hz:
        print("\n| head, line | test-like ± cluster SE (sel) | benchmark-equal | " + " | ".join(PARENTS)
              + " | mix/whole | R1 b / p | folds on | placebo test-like |\n|" + "---|" * 11)
        for n, v in hz.items():
            if n.startswith("_"):
                continue
            for ln_name, ln in v["lines"].items():
                pp = ln["tl_per_parent"]
                pl = v["placebo"].get(ln_name, {})
                sel = ln.get("tl_sel_se")
                print(f"| {n}, {ln_name} | {ln['tl']:+.5f} ± {ln['tl_cluster_se']:.5f}"
                      + ("" if sel is None else f" ({sel:.5f})") + f" | {ln['tl_benchmark_equal']:+.5f} | "
                      + " | ".join(f"{pp[q][0]:+.5f}" if q in pp else "" for q in PARENTS)
                      + f" | {ln['mix']:+.5f} | {ln['r1b']:+.5f} / {ln['r1p']:+.5f} | "
                      + ("" if ln["folds_on"] is None else f"{ln['folds_on']}/4")
                      + f" | {pl.get('tl', float('nan')):+.5f} |")
            print(f"  {n}: within-pair r {v.get('r_within_pair_tl')}, coverage {v['coverage_eval_items']}")
    if s.get("reference"):
        print("\nreference (honest difficulty degraded to r on the sampled items):")
        for r, t in s["reference"]["r"].items():
            print(f"  {r}: " + "  ".join(f"{n} {v['tl']:+.5f}" for n, v in t.items()))
    if s.get("verdict"):
        print("\nverdict:", json.dumps(s["verdict"], indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=("sample", "extract", "heads", "harness", "reference", "verdict", "show"))
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="extract even if another LLM process is alive")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--benchmarks", nargs="+", default=["matharena", "real_webagents", "multi_swebench",
                                                        "researchcodebench"])
    ap.add_argument("--rows", default=H.ROWS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boots", type=int, default=BOOT)
    ap.add_argument("--placebo", type=int, default=N_PLACEBO)
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--partial", action="store_true", help="heads on whatever is extracted so far (a smoke test)")
    args = ap.parse_args()
    {"sample": stage_sample, "extract": stage_extract, "heads": stage_heads, "harness": stage_harness,
     "reference": stage_reference, "verdict": stage_verdict, "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
