"""Fine-tuning an encoder: does Qwen3-Embedding-0.6B, adapted with low-rank
adapters to rank the items of a benchmark by difficulty, order the items of a
benchmark it never saw?

The frozen embedding carried nothing to an unseen benchmark (docs/findings.md,
"Neural embeddings do not carry difficulty to an unseen benchmark"), and every
frozen-feature head since was switched off by nested selection. The one thing
left untested is training the encoder itself, with a loss that asks only for
the order inside a benchmark. This is that test, run once, with the plan's
gate declared before any result (step 9 of the rethink plan; literature prior
about 10-15%: ADeLe's LoRA 8B lost out of distribution to a rubric, 0.692
against 0.747 AUROC).

Items and targets (--stage data, CPU; the tokenizer only). Unique items as
experiments/llm_features.py builds them (llm_features.unique_items: one row per
distinct visible text, with the item_ids it stands for). The text is
llmfeat.item_text (item_content, a newline, item_features), tokenized by the
embedding model's tokenizer and cut by llmfeat.embedding_ids to MAX_TOK tokens,
end-of-text included (head and tail around llmfeat.MARKER): the frozen
embedding's input rule with 512 tokens instead of 2,048.
  parents   the four multi-subject benchmarks. Target: the harness's honest
            difficulty (llm4b_close.honest_targets: Rasch b fitted without
            each of the five subject folds, averaged over the folds; higher =
            harder), averaged over the item_ids a text stands for. Reported
            beside it, never trained on: strong-tier difficulty, Rasch b over
            the pairs whose Rasch ability is at or above the parent's median
            (attempt_probe's definition).
  swe_rebench  training only (one subject, about 10 trials per item). Target:
            -logit((k + 0.5) / (n + 1)) from the item's k successes in n >=
            SWE_MIN_TRIALS trials. N_SWE items drawn at random (SEED).
Targets are standardised within benchmark over the training items. In every
benchmark IDV_FRAC of the items with a target (at most IDV_MAX, by a hash of
the key) are held out of every training run: an in-distribution check that the
fine-tuning learns anything at all.

Frozen lower layers (--stage lora-cache, the only language-model process). The
model is Qwen3-Embedding-0.6B in float32 on MPS. Its 28 decoder layers split
at L0 = 28 - K_TOP: layers 0..L0-1 stay frozen, so their output (the residual
stream entering layer L0) is computed once per item and cached as float16 in
data/finetune/cache/<benchmark>/NNNN.npy (gitignored). The same pass runs the
remaining layers, the final norm and last-token pooling (llmfeat's rule: the
end-of-text token's state, L2-normalised) to give the frozen embedding at 512
tokens. Batches are right-padded with no attention mask: attention is causal,
so no real token sees a pad. Shards are written atomically and --resume skips
the ones on disk. Check: against the stored 2,048-token embeddings
(data/features) on items short enough that both read the same tokens.

Fine-tuning (--stage lora). The top K_TOP = 6 layers get low-rank adapters,
written in plain torch (no peft): every linear map of attention and MLP (q, k,
v, o, gate, up, down), rank LORA_R = 16, alpha 32, A Kaiming-uniform and B
zero, so at step 0 the model is the frozen one. No adapter dropout: with
gradient checkpointing per layer on MPS the recomputation does not replay the
dropout masks (gradients with and without checkpointing differed by more than
their own size with dropout 0.05, and agreed exactly with none), so dropout
would give wrong gradients. A linear head reads the pooled, normalised
embedding. The cached states make a step cost the top six layers only.

Loss, per batch of BATCH = 16 items of one benchmark: RankNet over every pair
of the batch, softplus(-sign(z_i - z_j) (s_i - s_j)) weighted by |z_i - z_j|
clipped at PAIR_CLIP, plus MSE_W = 0.1 x the MSE between the batch-centred
scores and the batch-centred standardised targets (a benchmark's level never
matters: the harness standardises x within benchmark). Only the order inside a
benchmark is asked for.

Protocol. Four leave-one-benchmark-out folds over the four parents. For held-out
parent q, each of the other three parents v in turn is the INNER early-stopping
benchmark and the remaining two plus swe_rebench are trained on (12 runs):
  epoch 0   the frozen embedding with a ridge head (alpha from RIDGE_ALPHAS,
            chosen by Pearson on v), which also initialises the head;
  epochs 1..EPOCHS   an epoch draws up to CAP items per training benchmark
            (without replacement, reshuffled each epoch), in single-benchmark
            batches in random order; AdamW (adapters LR_LORA, head LR_HEAD,
            weight decay WD), WARMUP steps then linear decay, gradient norm
            clipped at CLIP_NORM.
The epoch with the best Pearson on v (0..EPOCHS, ties to the earlier) is kept:
early stopping never sees q. q's prediction is the mean over the three inner
models of their predictions standardised within q (PRIMARY, "finetuned").
Everything was fixed before a result was seen; nothing is tuned on q.

Baselines, all on the same items and targets (--stage eval):
  frozen_nested   epoch 0 of the same 12 runs: the frozen 512-token embedding
                  with the same data, the same nested choice and the same
                  ensembling. The paired control: all that differs is training.
  frozen_lobo512  ridge on the frozen 512-token embedding, fitted on all three
                  other parents plus swe_rebench, alpha by inner
                  leave-one-benchmark-out.
  frozen_lobo2048 the same on the stored 2,048-token embeddings (data/features;
                  no swe_rebench, which has none).
  emb_transfer    results/emb_transfer.json as published (in-sample Rasch
                  target, 2,048 tokens), quoted.
Held-out r: Pearson (and Spearman, and within item_features group) of x against
the target over the held-out parent's item_ids, with 95% intervals from a
bootstrap over item_features groups and over items (llm4b_close.corr_block);
a DerSimonian-Laird random-effects mean over the four parents with its
prediction interval; the paired difference finetuned - frozen_nested with a
group-bootstrap interval.

--stage harness. The out-of-fold predictions as covariates through
experiments/harness.py on its stored rows of the shipped hier (the recipe of
hidden_state_probe.stage_harness: x standardised within benchmark, the B0 term
on raw x, the nested transferred, per-pair, hybrid and B0 lines and the forced
ones; a placebo, x permuted within benchmark, for the primary).

--stage verdict (the plan's rule, declared before any result). GO only if the
primary has held-out Pearson r >= GO_R = 0.3 against honest difficulty on at
least GO_PARENTS = 3 of the 4 parents AND a nested test-like ALC difference
<= GO_ALC = -0.002 on the transferred or the per-pair line. Otherwise KILL.

The configuration (ridge head, adapters at 2e-4) was the one declared first. Two
alternatives were piloted for one epoch of one run, read on the idv items only
(never on a held-out parent): FT_HEAD_INIT=zero (a zero head) and
FT_LR_LORA=1e-3. Both lowered idv r (mean change -0.055 and -0.027, against
+0.004), so the declared one was kept. The two environment switches exist only
for that pilot; the results file records the values used (meta.config).

Run:  python experiments/finetune_encoder.py --stage data                 # 25 s, CPU, the tokenizer
      python experiments/finetune_encoder.py --stage lora-cache --resume  # 12 min, the LLM process
      nohup python experiments/finetune_encoder.py --stage lora --resume > LOG 2>&1 &
                              # 2 h 58 min for the 12 runs on an M1 Pro (MPS), one process
      python experiments/finetune_encoder.py --stage eval                 # 4 min, CPU, 1 GB
      python experiments/finetune_encoder.py --stage harness              # 4 min, CPU, 0.7 GB
      python experiments/finetune_encoder.py --stage verdict
      python experiments/finetune_encoder.py --stage show
The two language-model stages refuse to start while another language-model
process is alive (OTHER_LLM; their command lines match it too). Needs
data/<benchmark>/, data/features/<benchmark>/ (llm_features.py), the harness
rows (python experiments/harness.py --stage collect) and Qwen3-Embedding-0.6B
in the local Hugging Face cache (no download).
"""
import os
import sys

_LLM = any(a in ("lora-cache", "lora") for a in sys.argv)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4" if _LLM else "1")
if _LLM:
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import harness as H  # noqa: E402
from experiments import hidden_state_probe as HSP  # noqa: E402
from experiments import itemcov_eval as ICE  # noqa: E402
from experiments import llm4b_close as L4  # noqa: E402
from paiec import itemcov as IC  # noqa: E402
from paiec import llmfeat as F  # noqa: E402

OUT = os.path.join(ROOT, "results", "finetune_encoder.json")
FT_DIR = os.path.join(ROOT, "data", "finetune")
CACHE_DIR = os.path.join(FT_DIR, "cache")
RUN_DIR = os.path.join(FT_DIR, "runs")
FEAT_DIR = os.path.join(ROOT, "data", "features")
EMB_TRANSFER = os.path.join(ROOT, "results", "emb_transfer.json")
PARENTS = H.PARENTS
EXTRA = "swe_rebench"
BENCHES = PARENTS + (EXTRA,)
GROUP_KEY = ICE.GROUP_KEY

MAX_TOK = 512                   # end-of-text included
N_LAYERS = 28
K_TOP = 6                       # adapted layers
L0 = N_LAYERS - K_TOP           # the cached residual stream enters this layer
N_SWE = 1500
SWE_MIN_TRIALS = 3
IDV_FRAC = 0.15
IDV_MAX = 200
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.0   # see the docstring: checkpointing does not replay dropout on MPS
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LR_LORA, LR_HEAD, WD = float(os.environ.get("FT_LR_LORA", 2e-4)), 1e-3, 0.01
WARMUP = 20
EPOCHS = 3
BATCH = 16
MIN_BATCH = 4
CAP = 384                       # items per training benchmark per epoch
CLIP_NORM = 1.0
MSE_W = 0.1
PAIR_CLIP = 2.0
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)   # per mean eigenvalue of X'WX
SEED = 20260927
HEAD_INIT = os.environ.get("FT_HEAD_INIT", "ridge")   # 'ridge' (the frozen ridge) or 'zero'
CACHE_BUDGET = 8192             # padded tokens per batch
CACHE_MAX_BATCH = 32
SHARD_ITEMS = 256
SHARD_TOKENS = 65536
EVAL_BUDGET = 8192
EVAL_MAX_BATCH = 64
GO_R = 0.3
GO_PARENTS = 3
GO_ALC = -0.002
BOOT = 2000
N_PLACEBO = 3
PLACEBO_BOOTS = 200
PRIMARY = "finetuned"
HARNESS_COVS = ("finetuned", "finetuned_e3", "frozen_nested", "frozen_lobo512")
NESTED = ("transferred nested", "per-pair nested", "hybrid nested", "b0 nested")
RULE_LINES = ("transferred nested", "per-pair nested")
OTHER_LLM = HSP.OTHER_LLM
FORCED_ALL = L4.FORCED_ALL


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


write_json, read_json, atomic_write, _dur = HSP.write_json, HSP.read_json, HSP.atomic_write, HSP._dur
pear, ridge, weights, random_effects, _demean = HSP.pear, HSP.ridge, HSP.weights, HSP.random_effects, HSP._demean


def script_digest():
    return H.digest(["experiments/finetune_encoder.py"])


def config():
    """Everything that decides the numbers, fixed before any result."""
    return {"max_tok": MAX_TOK, "k_top": K_TOP, "l0": L0, "n_swe": N_SWE, "swe_min_trials": SWE_MIN_TRIALS,
            "idv_frac": IDV_FRAC, "idv_max": IDV_MAX, "lora": [LORA_R, LORA_ALPHA, LORA_DROPOUT],
            "targets": TARGET_MODULES, "lr": [LR_LORA, LR_HEAD], "wd": WD, "warmup": WARMUP, "epochs": EPOCHS,
            "batch": BATCH, "min_batch": MIN_BATCH, "cap": CAP, "clip": CLIP_NORM, "mse_w": MSE_W,
            "pair_clip": PAIR_CLIP, "ridge_alphas": RIDGE_ALPHAS, "seed": SEED, "head_init": HEAD_INIT,
            "go": {"r": GO_R, "parents": GO_PARENTS, "alc": GO_ALC}}


def _h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


# --- targets and items ----------------------------------------------------------------------

def strong_targets():
    """{parent: {item_id: b}}: Rasch difficulty over the pairs whose Rasch
    ability is at or above the parent's median (attempt_probe's strong tier)."""
    from paiec import data as D
    from paiec import official as O
    from paiec import testlike as T
    from paiec.rasch import rasch
    out, info = {}, {}
    for par in PARENTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = O.eligible(D.load_pairs([par]))
        si, ii, y, keys = [], [], [], {}
        for j, p in enumerate(ps):
            for x in p.responses:
                si.append(j)
                ii.append(keys.setdefault(x.item_key, len(keys)))
                y.append(x.label)
        th, _ = rasch(np.array(si), np.array(ii), np.array(y, float), len(ps), len(keys))
        med = float(np.median(th))
        strong = [p for p, t in zip(ps, th) if t >= med]
        out[par] = T.rasch(strong)
        info[par] = {"pairs": len(ps), "strong_pairs": len(strong), "items": len(out[par])}
    return out, info


def swe_counts():
    """{item_id: (successes, trials)} of swe_rebench's one subject."""
    import pandas as pd
    r = pd.read_parquet(os.path.join(ROOT, "data", EXTRA, "response.parquet"), columns=["item_id", "response"])
    r = r[r.response.isin([0.0, 1.0])]
    g = r.groupby(r.item_id.astype(str)).response.agg(["sum", "count"])
    return {str(i): (float(k), int(n)) for i, k, n in zip(g.index, g["sum"], g["count"])}


def stage_data(args):
    from transformers import AutoTokenizer
    from experiments import llm_features as LF
    t0 = time.time()
    honest_mean, honest_folds, tinfo = L4.honest_targets()
    log(f"honest targets in {time.time() - t0:.0f}s")
    strong, sinfo = strong_targets()
    log(f"strong-tier targets in {time.time() - t0:.0f}s: {sinfo}")
    swe = swe_counts()
    path, rev = LF.snapshot(F.EMB_REPO)
    tok = AutoTokenizer.from_pretrained(path)
    marker, tail = F.encode(tok, F.MARKER), F.appended_special(tok)
    meta, ids_all, info = {}, {}, {}
    for b in BENCHES:
        rows, n_rows = LF.unique_items(b)
        yh, ys = [], []
        for r in rows:
            ids = r["item_ids"]
            if b in PARENTS:
                h = [honest_mean[b][i] for i in ids if i in honest_mean[b]]
                s = [strong[b][i] for i in ids if i in strong[b]]
                yh.append(float(np.mean(h)) if h else float("nan"))
                ys.append(float(np.mean(s)) if s else float("nan"))
            else:
                kn = [swe[i] for i in ids if i in swe]
                k, n = sum(v[0] for v in kn), sum(v[1] for v in kn)
                yh.append(-math.log((k + 0.5) / (n - k + 0.5)) if n >= SWE_MIN_TRIALS else float("nan"))
                ys.append(float("nan"))
        yh, ys = np.array(yh), np.array(ys)
        keep = np.arange(len(rows))
        if b == EXTRA:
            ok = np.flatnonzero(np.isfinite(yh))
            keep = np.sort(np.random.default_rng(SEED).permutation(ok)[:N_SWE])
        rows = [rows[i] for i in keep]
        yh, ys = yh[keep], ys[keep]
        texts = [F.item_text(r["item"]) for r in rows]
        seqs, ntext, trunc = [], [], []
        for i0 in range(0, len(texts), 256):
            enc = tok(texts[i0:i0 + 256], add_special_tokens=False)["input_ids"]
            for e in enc:
                ids, n_text, tr = F.embedding_ids(e, tail, marker, MAX_TOK)
                seqs.append(np.array(ids, np.int32))
                ntext.append(n_text)
                trunc.append(bool(tr))
        keys = [r["key"] for r in rows]
        has = np.flatnonzero(np.isfinite(yh))
        order = sorted(has, key=lambda i: _h(keys[i] + ":idv"))
        n_idv = min(int(round(IDV_FRAC * len(has))), IDV_MAX)
        idv = np.zeros(len(rows), bool)
        idv[order[:n_idv]] = True
        meta[b] = {"keys": keys, "item_ids": [r["item_ids"] for r in rows], "y_honest": yh.tolist(),
                   "y_strong": ys.tolist(), "idv": idv.tolist(), "tokens": [len(s) for s in seqs],
                   "text_tokens": ntext, "truncated": trunc}
        ids_all[b] = seqs
        info[b] = {"unique": len(rows), "item_rows": n_rows, "with_target": int(len(has)), "idv": int(n_idv),
                   "tokens": int(sum(len(s) for s in seqs)), "truncated": int(sum(trunc)),
                   "median_text_tokens": float(np.median(ntext)) if ntext else 0.0}
        log(f"{b}: {info[b]}")
    os.makedirs(FT_DIR, exist_ok=True)
    arrays = {}
    for b, seqs in ids_all.items():
        arrays[f"{b}__ids"] = np.concatenate(seqs) if seqs else np.zeros(0, np.int32)
        arrays[f"{b}__off"] = np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64)
    atomic_write(os.path.join(FT_DIR, "tokens.npz"), lambda fh: np.savez(fh, **arrays))
    write_json(os.path.join(FT_DIR, "items.json"),
               {"meta": meta, "info": info, "config": config(), "tokenizer": {"repo": F.EMB_REPO, "revision": rev},
                "tail": tail, "marker_tokens": len(marker)})
    write_json(os.path.join(FT_DIR, "targets.json"),
               {"honest_mean": honest_mean, "honest_folds": honest_folds, "strong": strong,
                "info": {"honest": tinfo, "strong": sinfo}, "provenance": H.provenance()})
    log(f"data: {time.time() - t0:.0f}s -> {FT_DIR}")


def load_items():
    d = read_json(os.path.join(FT_DIR, "items.json"))
    if d is None:
        raise SystemExit("run --stage data first")
    return d


def load_tokens():
    with np.load(os.path.join(FT_DIR, "tokens.npz")) as z:
        out = {}
        for b in BENCHES:
            ids, off = z[f"{b}__ids"], z[f"{b}__off"]
            out[b] = [ids[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return out


# --- the frozen pass ----------------------------------------------------------------------

def refuse_if_other_llm(force):
    others = HSP.other_llm_processes()
    if others and not force:
        raise SystemExit("another language-model process is alive; at most one may run:\n" + "\n".join(others))


def shard_paths(b, s):
    d = os.path.join(CACHE_DIR, b)
    return os.path.join(d, f"{s:04d}.npy"), os.path.join(d, f"{s:04d}.npz")


def cache_plan(meta):
    return {b: F.plan(meta[b]["tokens"], meta[b]["keys"], SHARD_ITEMS, SHARD_TOKENS) for b in BENCHES}


def stage_cache(args):
    refuse_if_other_llm(args.force)
    import torch
    import transformers
    from transformers import AutoModel
    from experiments import llm_features as LF
    d = load_items()
    meta = d["meta"]
    toks = load_tokens()
    plan = cache_plan(meta)
    path, rev = LF.snapshot(F.EMB_REPO)
    cfg = {"model": F.EMB_REPO, "revision": rev, "dtype": "float32", "states": "float16", "l0": L0,
           "max_tok": MAX_TOK, "shard_items": SHARD_ITEMS, "shard_tokens": SHARD_TOKENS,
           "budget": CACHE_BUDGET, "max_batch": CACHE_MAX_BATCH, "device": args.device,
           "plan_digest": F.digest({b: [[meta[b]["keys"][i] for i in s] for s in p] for b, p in plan.items()})}
    chash = F.digest(cfg)
    mpath = os.path.join(CACHE_DIR, "manifest.json")
    man = read_json(mpath)
    if man is not None and not args.resume:
        raise SystemExit(f"{mpath} exists: pass --resume")
    if man is not None and man.get("config_hash") != chash:
        raise SystemExit("the cache was written for another config or plan; refusing to mix")
    if man is None:
        man = {"config": cfg, "config_hash": chash, "started": datetime.now().isoformat(timespec="seconds"),
               "env": {"torch": torch.__version__, "transformers": transformers.__version__}}
        write_json(mpath, man)
    dev = args.device
    torch.manual_seed(0)
    t0 = time.time()
    model = LF.load_weights(AutoModel, path, torch.float32, dev)
    layers, norm, rot, emb_w = model.layers, model.norm, model.rotary_emb, model.embed_tokens.weight
    from transformers import AutoTokenizer
    pad_id = AutoTokenizer.from_pretrained(path).pad_token_id
    log(f"model loaded in {time.time() - t0:.0f}s")
    todo = [(b, s) for b in BENCHES for s in range(len(plan[b])) if not os.path.exists(shard_paths(b, s)[1])]
    total_tok = sum(sum(meta[b]["tokens"][i] for i in plan[b][s]) for b, s in todo)
    done_tok, t1 = 0, time.time()
    for b, s in todo:
        idx = plan[b][s]
        seqs = [toks[b][i] for i in idx]
        lens = [len(x) for x in seqs]
        states = np.zeros((sum(lens), 1024), np.float16)
        embs = np.zeros((len(idx), 1024), np.float32)
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        with torch.inference_mode():
            for bt in F.batches(lens, CACHE_BUDGET, CACHE_MAX_BATCH):
                ids, _ = LF._pad([seqs[i] for i in bt], pad_id)
                ids = ids.to(dev)
                h = torch.nn.functional.embedding(ids, emb_w)
                pos = torch.arange(h.shape[1], device=dev)[None]
                pe = rot(h, pos)
                for li in range(N_LAYERS):
                    if li == L0:
                        hc = h.to(torch.float16).cpu().numpy()
                        for r, i in enumerate(bt):
                            states[off[i]:off[i + 1]] = hc[r, :lens[i]]
                    out = layers[li](h, attention_mask=None, position_ids=pos, position_embeddings=pe,
                                     use_cache=False)
                    h = out[0] if isinstance(out, tuple) else out
                h = norm(h)
                last = h[torch.arange(len(bt), device=dev), torch.tensor([lens[i] - 1 for i in bt], device=dev)]
                if not torch.isfinite(last).all():
                    raise FloatingPointError("non-finite embedding")
                e = torch.nn.functional.normalize(last.float(), dim=-1).cpu().numpy()
                embs[bt] = e
            if dev == "mps":
                torch.mps.empty_cache()
        if not np.isfinite(states.astype(np.float32)).all():
            raise FloatingPointError(f"{b} shard {s}: non-finite cached state (float16 overflow?)")
        p_npy, p_npz = shard_paths(b, s)
        atomic_write(p_npy, lambda fh: np.save(fh, states))
        atomic_write(p_npz, lambda fh: np.savez(fh, key=np.array([meta[b]["keys"][i] for i in idx]), off=off,
                                                emb=embs, lens=np.array(lens)))
        done_tok += off[-1]
        el = time.time() - t1
        rate = done_tok / max(el, 1e-9)
        left = (total_tok - done_tok) / max(rate, 1e-9)
        log(f"{b} shard {s + 1}/{len(plan[b])}: {len(idx)} items, {off[-1]} tokens; {rate:.0f} tok/s; "
            f"left {_dur(left)}, finish ~{(datetime.now() + timedelta(seconds=left)).strftime('%H:%M')}")
        man["rate_tok_per_s"] = rate
        man["updated"] = datetime.now().isoformat(timespec="seconds")
        write_json(mpath, man)
    man["complete"] = all(os.path.exists(shard_paths(b, s)[1]) for b in BENCHES for s in range(len(plan[b])))
    man["check_stored"] = check_stored(meta)
    log(f"check against the stored 2,048-token embeddings: {man['check_stored']}")
    write_json(mpath, man)
    log(f"cache: {_dur(time.time() - t0)}; complete={man['complete']}")


class Store:
    """The cached states (memory-mapped) and frozen embeddings, by (benchmark, key)."""

    def __init__(self, meta, benches=BENCHES):
        plan = cache_plan(meta)
        self.arrs, self.where, self.emb = [], {}, {}
        for b in benches:
            for s in range(len(plan[b])):
                p_npy, p_npz = shard_paths(b, s)
                if not os.path.exists(p_npz):
                    raise SystemExit(f"{b} shard {s} is not cached: run --stage lora-cache")
                arr = np.load(p_npy, mmap_mode="r")
                self.arrs.append(arr)
                with np.load(p_npz) as z:
                    keys, off, lens, emb = z["key"], z["off"], z["lens"], z["emb"]
                for r, k in enumerate(keys):
                    self.where[(b, str(k))] = (len(self.arrs) - 1, int(off[r]), int(lens[r]))
                    self.emb[(b, str(k))] = emb[r]

    def frozen(self, b, keys):
        return np.stack([self.emb[(b, k)] for k in keys]).astype(np.float64)

    def length(self, b, k):
        return self.where[(b, k)][2]

    def batch(self, b, keys, torch, dev):
        lens = [self.where[(b, k)][2] for k in keys]
        X = np.zeros((len(keys), max(lens), 1024), np.float16)
        for r, k in enumerate(keys):
            a, o, n = self.where[(b, k)]
            X[r, :n] = self.arrs[a][o:o + n]
        return (torch.from_numpy(X).to(dev).float(),
                torch.tensor(lens, device=dev, dtype=torch.long))


def check_stored(meta):
    """Cosine between this pass's frozen embedding and the stored 2,048-token one
    (data/features, float16) on the parents' items that fit both untruncated,
    so both read the same tokens."""
    st = Store(meta)
    out = {}
    for b in PARENTS:
        try:
            index, emb, _, _ = F.load(FEAT_DIR, b)
        except FileNotFoundError:
            continue
        pos = {k: i for i, k in enumerate(index["key"])}
        ks = [k for k, n, t in zip(meta[b]["keys"], meta[b]["text_tokens"], meta[b]["truncated"])
              if not t and k in pos and index["emb_tokens"].iloc[pos[k]] == n + 1 and (b, k) in st.emb]
        if not ks:
            continue
        a = st.frozen(b, ks)
        s = emb[[pos[k] for k in ks]].astype(np.float64)
        cos = (a * s).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(s, axis=1)
        out[b] = {"n": len(ks), "min": round(float(cos.min()), 6), "median": round(float(np.median(cos)), 6)}
    return out


# --- the adapted model ------------------------------------------------------------------------

def build_top_factory(path, dev):
    """(make(seed) -> a fresh adapted model on dev, cfg). The top K_TOP layers and
    the final norm come from the checkpoint (bf16 -> float32, as load_weights)."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights
    from transformers.models.qwen3 import modeling_qwen3 as Q
    from experiments import llm_features as LF
    import glob
    LF._no_sdpa_gqa()
    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "sdpa"
    if getattr(cfg, "use_sliding_window", False):
        raise ValueError("sliding-window layers are not supported")
    if cfg.num_hidden_layers != N_LAYERS:
        raise ValueError(f"{cfg.num_hidden_layers} layers, expected {N_LAYERS}")
    files = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        h = safe_open(f, framework="pt", device="cpu")
        for n in h.keys():
            files[n] = h
    prefix = "model." if "model.embed_tokens.weight" in files else ""
    pristine = {}
    for i in range(K_TOP):
        pre = f"{prefix}layers.{L0 + i}."
        for n in files:
            if n.startswith(pre):
                pristine[(i, n[len(pre):])] = files[n].get_tensor(n).float()
    norm_w = files[prefix + "norm.weight"].get_tensor(prefix + "norm.weight").float()

    class LoRALinear(torch.nn.Module):
        def __init__(self, base, gen):
            super().__init__()
            self.base = base
            A = torch.empty(LORA_R, base.in_features)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=gen)
            self.A = torch.nn.Parameter(A.to(base.weight.device))
            self.B = torch.nn.Parameter(torch.zeros(base.out_features, LORA_R, device=base.weight.device))
            self.scale = LORA_ALPHA / LORA_R
            self.drop = torch.nn.Dropout(LORA_DROPOUT)

        def forward(self, x):
            return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale

    def run_layer(layer, h, pos, cos, sin):
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
        return out[0] if isinstance(out, tuple) else out

    class Top(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            with no_init_weights(), torch.device(dev):
                self.layers = torch.nn.ModuleList([Q.Qwen3DecoderLayer(cfg, L0 + i) for i in range(K_TOP)])
                self.norm = Q.Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
                self.rotary = Q.Qwen3RotaryEmbedding(cfg)
                self.head = torch.nn.Linear(cfg.hidden_size, 1)
            with torch.no_grad():
                for i, lay in enumerate(self.layers):
                    prm = dict(lay.named_parameters())
                    if set(prm) != {n for (j, n) in pristine if j == i}:
                        raise KeyError(f"layer {L0 + i}: checkpoint and module disagree on parameter names")
                    for n, p in prm.items():
                        p.copy_(pristine[(i, n)])
                self.norm.weight.copy_(norm_w)
                self.head.weight.zero_()
                self.head.bias.zero_()
            for p in self.parameters():
                p.requires_grad_(False)
            gen = torch.Generator().manual_seed(seed)
            for lay in self.layers:
                for mod, names in ((lay.self_attn, TARGET_MODULES[:4]), (lay.mlp, TARGET_MODULES[4:])):
                    for n in names:
                        setattr(mod, n, LoRALinear(getattr(mod, n), gen))
            for p in self.head.parameters():
                p.requires_grad_(True)

        def adapters(self):
            return [p for n, p in self.named_parameters() if n.endswith((".A", ".B"))]

        def trainable_state(self):
            return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

        def load_trainable(self, state):
            prm = dict(self.named_parameters())
            with torch.no_grad():
                for n, v in state.items():
                    prm[n].copy_(v.to(prm[n].device))

        def embed(self, h, lengths):
            from torch.utils.checkpoint import checkpoint
            B, L, _ = h.shape
            pos = torch.arange(L, device=h.device)[None]
            cos, sin = self.rotary(h, pos)
            for lay in self.layers:
                if self.training:
                    h = checkpoint(run_layer, lay, h, pos, cos, sin, use_reentrant=False)
                else:
                    h = run_layer(lay, h, pos, cos, sin)
            h = self.norm(h)
            last = h[torch.arange(B, device=h.device), lengths - 1]
            return torch.nn.functional.normalize(last, dim=-1)

        def forward(self, h, lengths):
            return self.head(self.embed(h, lengths))[:, 0]

    for (i, n), v in pristine.items():
        if not torch.isfinite(v).all():
            raise RuntimeError(f"non-finite weight in layer {L0 + i} {n}")
    return Top, cfg


def pair_loss(s, z, torch):
    """RankNet over the batch's pairs, weighted by |z_i - z_j| (clipped), plus
    MSE_W x the MSE of the batch-centred scores against the batch-centred
    targets. Returns (loss, rank part, mse part, pairwise accuracy)."""
    d = s[:, None] - s[None, :]
    t = z[:, None] - z[None, :]
    iu = torch.triu(torch.ones_like(t, dtype=torch.bool), diagonal=1)
    w = torch.clamp(t.abs(), max=PAIR_CLIP) * iu * (t.abs() > 0)
    rank = (w * torch.nn.functional.softplus(-torch.sign(t) * d)).sum() / w.sum().clamp_min(1e-9)
    mse = (((s - s.mean()) - (z - z.mean())) ** 2).mean()
    acc = ((torch.sign(d) == torch.sign(t)).float() * (w > 0)).sum() / (w > 0).sum().clamp_min(1)
    return rank + MSE_W * mse, rank, mse, acc


def predict_keys(model, store, b, keys, torch, dev):
    """Scores for keys of benchmark b, in order (eval mode, batches by length)."""
    if not keys:
        return np.zeros(0)
    model.eval()
    order = sorted(range(len(keys)), key=lambda i: (store.length(b, keys[i]), keys[i]))
    lens = [store.length(b, keys[i]) for i in order]
    out = np.zeros(len(keys))
    with torch.no_grad():
        for bt in F.batches(lens, EVAL_BUDGET, EVAL_MAX_BATCH):
            ks = [keys[order[j]] for j in bt]
            X, L = store.batch(b, ks, torch, dev)
            s = model(X, L).float().cpu().numpy().astype(np.float64)
            for j, v in zip(bt, s):
                out[order[j]] = v
    if dev == "mps":
        torch.mps.empty_cache()
    return out


def run_sets(meta, q, v):
    """(training {bench: (keys, z)}, v's keys and targets, q's keys, idv {bench:
    (keys, y)}). Targets standardised within benchmark over the training keys."""
    train, idv = {}, {}
    for b in [p for p in PARENTS if p not in (q, v)] + [EXTRA]:
        m = meta[b]
        y = np.array(m["y_honest"], float)
        tr = [i for i in range(len(y)) if np.isfinite(y[i]) and not m["idv"][i]]
        mu, sd = y[tr].mean(), y[tr].std()
        train[b] = ([m["keys"][i] for i in tr], (y[tr] - mu) / sd)
        iv = [i for i in range(len(y)) if m["idv"][i]]
        idv[b] = ([m["keys"][i] for i in iv], y[iv])
    yv = np.array(meta[v]["y_honest"], float)
    vi = [i for i in range(len(yv)) if np.isfinite(yv[i])]
    return train, ([meta[v]["keys"][i] for i in vi], yv[vi]), list(meta[q]["keys"]), idv


def frozen_ridge(store, train, vset):
    """Ridge on the frozen embedding (centred within benchmark, items weighted
    1 / benchmark count) with alpha by Pearson on v. Returns (w, alpha, {alpha: r_v})."""
    Xs, ys, bs = [], [], []
    for b, (ks, z) in train.items():
        X = store.frozen(b, ks)
        Xs.append(X - X.mean(0))
        ys.append(z)
        bs += [b] * len(ks)
    X, y, bs = np.vstack(Xs), np.concatenate(ys), np.array(bs, object)
    w = weights(bs)
    Ev = store.frozen(vset[2], vset[0])
    crit, best = {}, None
    for a in RIDGE_ALPHAS:
        beta = ridge(X, y, w, a)
        crit[a] = pear(Ev @ beta, vset[1])
        if best is None or crit[a] > crit[best[0]]:
            best = (a, beta)
    return best[1], best[0], crit


def run_name(q, v):
    return f"{q}__{v}"


def train_run(q, v, meta, store, Top, torch, dev, run_dir, max_steps=None, check=None):
    """One inner run: frozen ridge warm start (epoch 0), EPOCHS epochs of
    adapter training, predictions for q, v and the idv items after each epoch.
    Checkpoints after every epoch; resumes from the last one."""
    name = run_name(q, v)
    ck_path = os.path.join(run_dir, name + ".ckpt.pt")
    train, vset, qkeys, idv = run_sets(meta, q, v)
    vset = (vset[0], vset[1], v)
    w0, alpha, crit = frozen_ridge(store, train, vset)
    seed = SEED + _h(name) % 100000
    model = Top(seed)
    if HEAD_INIT == "ridge":
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor(w0, dtype=torch.float32)[None])
    opt = torch.optim.AdamW([{"params": model.adapters(), "lr": LR_LORA},
                             {"params": list(model.head.parameters()), "lr": LR_HEAD}], weight_decay=WD)
    steps_epoch = sum(len(chunks(len(ks), min(len(ks), CAP))) for ks, _ in train.values())
    total = steps_epoch * EPOCHS if max_steps is None else max_steps

    def lr_at(step):
        return min(1.0, (step + 1) / WARMUP) * max(0.0, 1.0 - step / max(total, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    # epoch 0: the frozen embedding and the ridge head
    E = {b: store.frozen(b, ks) for b, (ks, _) in idv.items()}
    preds = {"q": [store.frozen(q, qkeys) @ w0], "v": [store.frozen(v, vset[0]) @ w0],
             "idv": {b: [E[b] @ w0] for b in idv}}
    curve = [{"epoch": 0, "r_v": pear(preds["v"][0], vset[1]),
              "r_idv": {b: pear(preds["idv"][b][0], idv[b][1]) for b in idv}}]
    check_out = None
    if check is not None:
        # the adapted model at step 0 is the frozen one: the same scores from the cached states
        ks = vset[0][:check]
        got = predict_keys(model, store, v, ks, torch, dev)
        ref = store.frozen(v, ks) @ w0
        check_out = {"n": len(ks), "max_abs_diff": float(np.abs(got - ref).max()),
                     "ref_sd": float(ref.std()), "corr": pear(got, ref)}
        log(f"{name}: step-0 check {check_out}")
    start = 1
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_trainable(ck["trainable"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        preds, curve, start = ck["preds"], ck["curve"], ck["epoch"] + 1
        check_out = ck.get("check", check_out)
        log(f"{name}: resumed after epoch {ck['epoch']}")
    t0 = time.time()
    step = (start - 1) * steps_epoch
    for ep in range(start, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng([SEED, _h(name) % 100000, ep])
        torch.manual_seed(SEED + 1000 * ep + _h(name) % 1000)
        batches = []
        for b, (ks, z) in train.items():
            pick = rng.permutation(len(ks))[:min(len(ks), CAP)]
            for c in chunks(len(ks), len(pick)):
                batches.append((b, [ks[pick[j]] for j in c], z[pick[c]]))
        batches = [batches[i] for i in rng.permutation(len(batches))]
        stats = {"loss": [], "rank": [], "mse": [], "acc": [], "tokens": 0}
        te = time.time()
        for bi, (b, ks, z) in enumerate(batches):
            if max_steps is not None and step >= max_steps:
                break
            X, L = store.batch(b, ks, torch, dev)
            s = model(X, L)
            loss, rk, ms, acc = pair_loss(s, torch.tensor(z, dtype=torch.float32, device=dev), torch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP_NORM)
            opt.step()
            sched.step()
            step += 1
            stats["loss"].append(float(loss))
            stats["rank"].append(float(rk))
            stats["mse"].append(float(ms))
            stats["acc"].append(float(acc))
            stats["tokens"] += int(X.shape[0] * X.shape[1])
            if (bi + 1) % 25 == 0:
                el = time.time() - te
                log(f"{name} ep {ep} step {bi + 1}/{len(batches)} loss {np.mean(stats['loss'][-25:]):.4f} "
                    f"acc {np.mean(stats['acc'][-25:]):.3f} ({el / (bi + 1):.2f}s/step)")
        train_s = time.time() - te
        te = time.time()
        # free the last training step's tensors before evaluating (MPS ran out of memory in evaluation)
        X = L = s = loss = rk = ms = acc = None
        opt.zero_grad(set_to_none=True)
        import gc
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
        preds["q"].append(predict_keys(model, store, q, qkeys, torch, dev))
        preds["v"].append(predict_keys(model, store, v, vset[0], torch, dev))
        for b in idv:
            preds["idv"][b].append(predict_keys(model, store, b, idv[b][0], torch, dev))
        curve.append({"epoch": ep, "r_v": pear(preds["v"][-1], vset[1]),
                      "r_idv": {b: pear(preds["idv"][b][-1], idv[b][1]) for b in idv},
                      "train_loss": float(np.mean(stats["loss"])) if stats["loss"] else None,
                      "train_rank": float(np.mean(stats["rank"])) if stats["rank"] else None,
                      "train_mse": float(np.mean(stats["mse"])) if stats["mse"] else None,
                      "train_pair_acc": float(np.mean(stats["acc"])) if stats["acc"] else None,
                      "steps": len(stats["loss"]), "padded_tokens": stats["tokens"],
                      "train_s": round(train_s, 1), "eval_s": round(time.time() - te, 1)})
        log(f"{name} epoch {ep}: r_v {curve[-1]['r_v']:+.4f} (epoch 0 {curve[0]['r_v']:+.4f}), idv "
            + " ".join(f"{b[:10]} {r:+.3f}" for b, r in curve[-1]["r_idv"].items())
            + f", pair acc {curve[-1]['train_pair_acc']}, train {train_s:.0f}s eval {curve[-1]['eval_s']:.0f}s")
        ck = {"trainable": model.trainable_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "preds": preds, "curve": curve, "epoch": ep, "check": check_out}
        tmp = ck_path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ck_path)
    best = max(range(len(curve)), key=lambda e: (round(curve[e]["r_v"], 10), -e))
    arrays = {"q_keys": np.array(qkeys), "q_pred": np.array(preds["q"]), "v_keys": np.array(vset[0]),
              "v_pred": np.array(preds["v"]), "v_y": vset[1]}
    for b in idv:
        arrays[f"idv__{b}__keys"] = np.array(idv[b][0])
        arrays[f"idv__{b}__pred"] = np.array(preds["idv"][b])
        arrays[f"idv__{b}__y"] = idv[b][1]
    atomic_write(os.path.join(run_dir, name + ".npz"), lambda fh: np.savez(fh, **arrays))
    write_json(os.path.join(run_dir, name + ".json"),
               {"q": q, "v": v, "train": {b: len(ks) for b, (ks, _) in train.items()}, "ridge_alpha": alpha,
                "ridge_crit": {str(a): c for a, c in crit.items()}, "curve": curve, "best_epoch": best,
                "steps_per_epoch": steps_epoch, "seed": seed, "step0_check": check_out,
                "max_steps": max_steps, "wall_s": round(time.time() - t0, 1), "config": config(),
                "script_digest": script_digest()})
    if os.path.exists(ck_path) and max_steps is None:
        os.remove(ck_path)
    del model, opt, sched
    return best, curve


def chunks(n, m):
    """Batches of BATCH over the first m of n shuffled positions; a last batch
    below MIN_BATCH is dropped."""
    out = [list(range(i, min(i + BATCH, m))) for i in range(0, m, BATCH)]
    return [c for c in out if len(c) >= MIN_BATCH]


def stage_lora(args):
    refuse_if_other_llm(args.force)
    import torch
    from experiments import llm_features as LF
    d = load_items()
    meta = d["meta"]
    run_dir = args.run_dir or RUN_DIR
    os.makedirs(run_dir, exist_ok=True)
    store = Store(meta)
    path, rev = LF.snapshot(F.EMB_REPO)
    Top, cfg = build_top_factory(path, args.device)
    runs = [(q, v) for q in PARENTS for v in PARENTS if v != q]
    if args.runs:
        runs = [tuple(r.split(":")) for r in args.runs]
    todo = [r for r in runs if not os.path.exists(os.path.join(run_dir, run_name(*r) + ".json"))]
    log(f"lora: {len(todo)} of {len(runs)} runs to do -> {run_dir}")
    t0 = time.time()
    for i, (q, v) in enumerate(todo):
        t1 = time.time()
        best, curve = train_run(q, v, meta, store, Top, torch, args.device, run_dir, max_steps=args.max_steps,
                                check=64 if i == 0 else None)
        # free the finished run's model and optimizer before the next one (without this the MPS pool
        # grew run by run and the eighth run of the first pass ran out of memory)
        import gc
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
            log(f"MPS allocated after the run: {torch.mps.current_allocated_memory() / 1e9:.2f} GB, "
                f"driver {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
        el = time.time() - t0
        left = el / (i + 1) * (len(todo) - i - 1)
        log(f"run {run_name(q, v)} done in {_dur(time.time() - t1)}: best epoch {best}, r_v "
            + " ".join(f"{c['r_v']:+.3f}" for c in curve)
            + f"; {len(todo) - i - 1} left, ~{_dur(left)}, finish "
              f"~{(datetime.now() + timedelta(seconds=left)).strftime('%H:%M')}")
    log(f"lora: {_dur(time.time() - t0)}")


# --- evaluation (CPU) ---------------------------------------------------------------------------

def zs(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def load_runs(run_dir=RUN_DIR):
    out = {}
    for q in PARENTS:
        for v in PARENTS:
            if v == q:
                continue
            p = os.path.join(run_dir, run_name(q, v))
            if not os.path.exists(p + ".json"):
                raise SystemExit(f"run {run_name(q, v)} is missing: run --stage lora")
            with np.load(p + ".npz") as z:
                out[(q, v)] = {"json": read_json(p + ".json"), **{k: z[k] for k in z.files}}
    return out


def assemble(runs):
    """{covariate: {parent: {key: x}}} from the 12 runs: the primary
    (early-stopped on v, mean over v of predictions standardised within q),
    each fixed epoch (epoch 0 = frozen_nested) and the best-on-v epochs."""
    covs = {"finetuned": {}, "frozen_nested": {}, **{f"finetuned_e{e}": {} for e in range(1, EPOCHS + 1)}}
    for q in PARENTS:
        rs = [runs[(q, v)] for v in PARENTS if v != q]
        keys = [str(k) for k in rs[0]["q_keys"]]
        for r in rs:
            if [str(k) for k in r["q_keys"]] != keys:
                raise ValueError(f"{q}: runs disagree on q's keys")
        best = np.mean([zs(r["q_pred"][r["json"]["best_epoch"]]) for r in rs], 0)
        covs["finetuned"][q] = dict(zip(keys, best))
        covs["frozen_nested"][q] = dict(zip(keys, np.mean([zs(r["q_pred"][0]) for r in rs], 0)))
        for e in range(1, EPOCHS + 1):
            covs[f"finetuned_e{e}"][q] = dict(zip(keys, np.mean([zs(r["q_pred"][e]) for r in rs], 0)))
    return covs


def frozen_lobo(meta, emb_of, benches_train_extra):
    """Ridge on frozen embeddings fitted on the other three parents (+ extra
    units), alpha by inner leave-one-benchmark-out (mean Pearson over the
    three), all keys with a target (idv included). emb_of(b, keys) -> X.
    Returns ({parent: {key: x}}, {parent: fold info})."""
    out, info = {}, {}

    def data(bs):
        Xs, ys, lab = [], [], []
        for b in bs:
            m = meta[b]
            y = np.array(m["y_honest"], float)
            ks = [k for k, t in zip(m["keys"], y) if np.isfinite(t)]
            X = emb_of(b, ks)
            if X is None:
                continue
            yy = y[np.isfinite(y)]
            Xs.append(X - X.mean(0))
            ys.append((yy - yy.mean()) / yy.std())
            lab += [b] * len(ks)
        return np.vstack(Xs), np.concatenate(ys), np.array(lab, object)

    for q in PARENTS:
        others = [p for p in PARENTS if p != q]
        crit = {}
        for a in RIDGE_ALPHAS:
            rs = []
            for v in others:
                X, y, lab = data([p for p in others if p != v] + list(benches_train_extra))
                beta = ridge(X, y, weights(lab), a)
                m = meta[v]
                yv = np.array(m["y_honest"], float)
                ks = [k for k, t in zip(m["keys"], yv) if np.isfinite(t)]
                rs.append(pear(emb_of(v, ks) @ beta, yv[np.isfinite(yv)]))
            crit[a] = float(np.mean(rs))
        a = max(RIDGE_ALPHAS, key=lambda x: (crit[x], -RIDGE_ALPHAS.index(x)))
        X, y, lab = data(others + list(benches_train_extra))
        beta = ridge(X, y, weights(lab), a)
        ks = list(meta[q]["keys"])
        Xq = emb_of(q, ks)
        out[q] = dict(zip(ks, Xq @ beta))
        info[q] = {"alpha": a, "inner_mean_r": crit}
    return out, info


def item_rows(meta, targets, q, xmap):
    """Rows per item_id of parent q with an honest target: x (the item's text's
    prediction), honest and strong-tier difficulty, group, log length,
    text-bearing."""
    items = ICE.load_items(q)
    hm, st = targets["honest_mean"][q], targets["strong"][q]
    tb = L4.text_bearing(items)
    R = {c: [] for c in ("iid", "x", "y", "ys", "g", "L", "text")}
    for k, ids in zip(meta[q]["keys"], meta[q]["item_ids"]):
        if k not in xmap:
            continue
        for iid in ids:
            if iid not in hm:
                continue
            it = items[iid]
            for c, val in (("iid", iid), ("x", xmap[k]), ("y", hm[iid]), ("ys", st.get(iid, np.nan)),
                           ("g", IC.features(it).get(GROUP_KEY[q], "")), ("L", IC.log_length(it)),
                           ("text", tb[iid])):
                R[c].append(val)
    return {c: np.array(v, object if c in ("iid", "g") else float) for c, v in R.items()}


def paired_diff(x1, x2, y, g, boots, seed):
    """Pearson(x1, y) - Pearson(x2, y) with a 95% interval from a bootstrap over
    groups (each drawn group a group of its own)."""
    est = pear(x1, y) - pear(x2, y)
    ug, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(ug))]
    rng = np.random.default_rng(seed)
    bs = []
    if len(ug) > 1:
        for _ in range(boots):
            ix = np.concatenate([members[j] for j in rng.integers(0, len(ug), len(ug))])
            bs.append(pear(x1[ix], y[ix]) - pear(x2[ix], y[ix]))
    ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)] if bs else None
    return {"est": round(float(est), 4), "ci_group": ci}


def stage_eval(args):
    t0 = time.time()
    d = load_items()
    meta = d["meta"]
    targets = read_json(os.path.join(FT_DIR, "targets.json"))
    runs = load_runs(args.run_dir or RUN_DIR)
    covs = assemble(runs)
    store = Store(meta)
    covs["frozen_lobo512"], lobo512 = frozen_lobo(meta, lambda b, ks: store.frozen(b, ks), (EXTRA,))
    stored = {}
    for b in PARENTS:
        index, emb, _, _ = F.load(FEAT_DIR, b)
        pos = {k: i for i, k in enumerate(index["key"])}
        stored[b] = (pos, emb)

    def emb2048(b, ks):
        if b not in stored:
            return None
        pos, emb = stored[b]
        return emb[[pos[k] for k in ks]].astype(np.float64)

    covs["frozen_lobo2048"], lobo2048 = frozen_lobo(meta, emb2048, ())
    log(f"eval: covariates assembled in {time.time() - t0:.0f}s")
    res = {"per_cov": {}, "folds": {"frozen_lobo512": lobo512, "frozen_lobo2048": lobo2048}}
    rows_all = {}
    for name, cmap in covs.items():
        per, pr, prs, ns = {}, [], [], []
        for i, q in enumerate(PARENTS):
            R = item_rows(meta, targets, q, cmap[q])
            rows_all[(name, q)] = R
            cb = L4.corr_block(R["x"], R["y"], R["g"], R["L"], args.boots, seed=97 * i + 11)
            ok = np.isfinite(R["ys"])
            cs = L4.corr_block(R["x"][ok], R["ys"][ok], R["g"][ok], R["L"][ok], args.boots, seed=97 * i + 12)
            ent = {"honest": cb, "strong": cs}
            if q == "matharena":
                t = R["text"] > 0
                ent["honest_text_bearing"] = L4.corr_block(R["x"][t], R["y"][t], R["g"][t], R["L"][t],
                                                           args.boots, seed=97 * i + 13)
            per[q] = ent
            pr.append(cb["pearson"]["est"])
            prs.append(cs["pearson"]["est"])
            ns.append(cb["n"])
        res["per_cov"][name] = {
            "per_parent": per, "pearson": dict(zip(PARENTS, pr)), "pearson_strong": dict(zip(PARENTS, prs)),
            "mean_pearson": round(float(np.mean(pr)), 4), "mean_pearson_strong": round(float(np.mean(prs)), 4),
            "parents_r_at_least": int(sum(r >= GO_R for r in pr)), "positive_parents": int(sum(r > 0 for r in pr)),
            "random_effects": random_effects(pr, ns), "random_effects_strong": random_effects(prs, ns)}
        log(f"{name:16s} r " + "  ".join(f"{q[:10]} {r:+.3f}" for q, r in zip(PARENTS, pr))
            + f" | mean {np.mean(pr):+.3f}, >= {GO_R}: {sum(r >= GO_R for r in pr)}/4 | strong "
            + " ".join(f"{r:+.3f}" for r in prs))
    # paired: finetuned (and each epoch) minus frozen_nested
    res["paired_vs_frozen_nested"] = {}
    for name in ["finetuned"] + [f"finetuned_e{e}" for e in range(1, EPOCHS + 1)]:
        res["paired_vs_frozen_nested"][name] = {}
        for i, q in enumerate(PARENTS):
            a, b = rows_all[(name, q)], rows_all[("frozen_nested", q)]
            if list(a["iid"]) != list(b["iid"]):
                raise ValueError("rows disagree")
            res["paired_vs_frozen_nested"][name][q] = paired_diff(a["x"], b["x"], a["y"], a["g"], args.boots,
                                                                  seed=31 * i + 7)
    # inner curves
    inner = {}
    for (q, v), r in runs.items():
        c = r["json"]["curve"]
        inner[run_name(q, v)] = {"best_epoch": r["json"]["best_epoch"], "ridge_alpha": r["json"]["ridge_alpha"],
                                 "r_v": [x["r_v"] for x in c], "r_idv": [x["r_idv"] for x in c],
                                 "train_pair_acc": [x.get("train_pair_acc") for x in c],
                                 "train_loss": [x.get("train_loss") for x in c],
                                 "wall_s": r["json"]["wall_s"], "step0_check": r["json"].get("step0_check")}
    gain_v = np.array([[x - v["r_v"][0] for x in v["r_v"][1:]] for v in inner.values()])
    gain_idv = {}
    for v in inner.values():
        for e in range(1, EPOCHS + 1):
            for b, x in v["r_idv"][e].items():
                gain_idv.setdefault(b, {}).setdefault(e, []).append(x - v["r_idv"][0][b])
    res["inner"] = {"runs": inner,
                    "mean_gain_r_v_by_epoch": gain_v.mean(0).round(4).tolist(),
                    "runs_improving_v_by_epoch": (gain_v > 0).sum(0).tolist(),
                    "best_epoch_counts": {e: int(sum(v["best_epoch"] == e for v in inner.values()))
                                          for e in range(EPOCHS + 1)},
                    "mean_idv_gain_by_epoch": {b: {e: round(float(np.mean(v)), 4) for e, v in d_.items()}
                                               for b, d_ in gain_idv.items()},
                    "mean_idv_r_by_epoch": {b: [round(float(np.mean([v["r_idv"][e][b] for v in inner.values()
                                                                        if b in v["r_idv"][e]])), 4)
                                                for e in range(EPOCHS + 1)] for b in BENCHES}}
    et = read_json(EMB_TRANSFER) or {}
    lb = (et.get("lobo") or {}).get("emb_ridge_centred") or {}
    res["emb_transfer_published"] = {q: {"pearson": lb[q].get("pearson"), "ci_group": lb[q].get("pearson_ci_groupboot"),
                                         "n": lb[q].get("n")} for q in PARENTS if q in lb}
    oof = {name: {iid: float(cmap[q][k]) for q in PARENTS for k, ids in zip(meta[q]["keys"], meta[q]["item_ids"])
                  if k in cmap[q] for iid in ids} for name, cmap in covs.items()}
    write_json(os.path.join(FT_DIR, "oof.json"), oof)
    cm = read_json(os.path.join(CACHE_DIR, "manifest.json")) or {}
    res["meta"] = {**H.provenance(), "script_digest": script_digest(), "config": config(), "data": d["info"],
                   "cache": {k: cm.get(k) for k in ("config", "rate_tok_per_s", "complete", "check_stored",
                                                    "started", "updated")},
                   "run_wall_s": {run_name(*k): r["json"]["wall_s"] for k, r in runs.items()},
                   "primary": PRIMARY,
                   "target": "honest Rasch difficulty (harness.oracle_maps), mean over 5 subject folds; strong-tier "
                             "Rasch b (pairs with ability >= the parent's median) reported, never trained on"}
    st = H.load_json(args.out) or {}
    st["eval"] = res
    st["meta"] = res.pop("meta")
    H.save_json(args.out, st)
    log(f"eval: {time.time() - t0:.0f}s -> {args.out}")


# --- harness, verdict, show ---------------------------------------------------------------------

def stage_harness(args):
    t0 = time.time()
    oof = read_json(os.path.join(FT_DIR, "oof.json"))
    if not oof:
        raise SystemExit("run --stage eval first")
    targets = read_json(os.path.join(FT_DIR, "targets.json"))
    honest = targets["honest_folds"]
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    items_bench = H.benchmark_items(rows, keys)
    with open(os.path.join(FT_DIR, "oof.json"), "rb") as fh:
        odig = hashlib.sha256(fh.read()).hexdigest()[:16]
    prov = H.provenance()
    state = H.load_json(args.out) or {}
    res = state.get("harness", {})
    if res.get("_meta", {}).get("oof_digest") != odig or res.get("_meta", {}).get("lib_digest") != prov["lib_digest"]:
        res = {}
    res["_meta"] = {"oof_digest": odig, "lib_digest": prov["lib_digest"], "rows_lib_digest": H.rows_digest(rows),
                    "forced": [H.cname(c) for c in FORCED_ALL], "n_placebo": args.placebo,
                    "placebo_boots": PLACEBO_BOOTS, "boots": H.BOOTS,
                    "input": "out-of-fold predictions, standardised within benchmark (harness.eval_covariate); "
                             "B0 term on raw x"}
    log(f"harness: rows in {time.time() - t0:.0f}s")
    for name in (args.covs or HARNESS_COVS):
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
        n_pl = args.placebo if name == PRIMARY else 0
        plac = {}
        for s in range(n_pl):
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
        HSP._save_part(args.out, "harness", res)
        a = entry["lines"]
        log(f"{name}: {entry['wall_s']}s  r_within_pair {entry.get('r_within_pair_tl')}  " + "  ".join(
            f"{n.split()[0]} {a[n]['tl']:+.5f} (on {a[n]['folds_on']})" for n in NESTED)
            + "  forced: " + "  ".join(f"{n} {a[n]['tl']:+.5f}" for n in a if n.endswith("(forced)")))
    HSP._save_part(args.out, "harness", res)
    log(f"harness: {time.time() - t0:.0f}s -> {args.out}")


def stage_verdict(args):
    st = H.load_json(args.out)
    if not st or "eval" not in st or "harness" not in st:
        raise SystemExit("run --stage eval and --stage harness first")
    per = {}
    for name, e in st["eval"]["per_cov"].items():
        hz = st["harness"].get(name)
        alc = None if not hz else {n: hz["lines"][n]["tl"] for n in hz["lines"]}
        gate = None if not hz else {n: hz["lines"][n]["gate_pass"] for n in NESTED if n in hz["lines"]}
        alc_prong = None if alc is None else any(alc[n] <= GO_ALC for n in RULE_LINES)
        r_prong = bool(e["parents_r_at_least"] >= GO_PARENTS)
        per[name] = {"pearson": e["pearson"], "parents_r_at_least": e["parents_r_at_least"],
                     "mean_pearson": e["mean_pearson"], "r_prong": r_prong, "alc_tl": alc,
                     "alc_prong": alc_prong, "harness_gate_pass": gate,
                     "go": bool(r_prong and alc_prong)}
    p = per[PRIMARY]
    st["verdict"] = {
        "rule": (f"GO only if the primary ({PRIMARY}) has held-out Pearson r >= {GO_R} against honest difficulty "
                 f"on >= {GO_PARENTS} of 4 parents AND a nested test-like ALC difference <= {GO_ALC} on "
                 f"{' or '.join(RULE_LINES)}; otherwise KILL"),
        "covariates": per, "call": "GO" if p["go"] else "KILL",
        "script_digest": script_digest()}
    H.save_json(args.out, st)
    print(json.dumps(st["verdict"], indent=1))


def _c(v, nd=2):
    if not v or v.get("est") is None:
        return ""
    ci = v.get("ci_group") or v.get("ci_item")
    return f"{v['est']:+.{nd}f}" + ("" if not ci else f" [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]")


def stage_show(args):
    s = H.load_json(args.out)
    m = s.get("meta", {})
    print("data:", json.dumps(m.get("data")), "\ncache:", json.dumps((m.get("cache") or {}).get("check_stored")),
          (m.get("cache") or {}).get("rate_tok_per_s"))
    ev = s.get("eval", {})
    if ev:
        print("\nheld-out Pearson vs honest difficulty [group CI]\n| covariate | " + " | ".join(PARENTS)
              + " | mean | >= 0.3 | RE mean [PI] |\n|" + "---|" * 8)
        for n, e in ev["per_cov"].items():
            print(f"| {n} | " + " | ".join(_c(e["per_parent"][q]["honest"].get("pearson")) for q in PARENTS)
                  + f" | {e['mean_pearson']:+.3f} | {e['parents_r_at_least']}/4 | {e['random_effects']['mean']:+.3f} "
                    f"{e['random_effects']['prediction_interval']} |")
        pub = ev.get("emb_transfer_published", {})
        print("| emb_transfer (published, in-sample target) | " + " | ".join(
            f"{pub[q]['pearson']:+.2f} [{pub[q]['ci_group'][0]:+.2f}, {pub[q]['ci_group'][1]:+.2f}]"
            if q in pub and pub[q].get("ci_group") else "" for q in PARENTS) + " | | | |")
        for stat in ("spearman", "pearson_within", "partial_pearson"):
            print(f"\n{stat} vs honest\n| covariate | " + " | ".join(PARENTS) + " |\n|" + "---|" * 5)
            for n, e in ev["per_cov"].items():
                print(f"| {n} | " + " | ".join(_c(e["per_parent"][q]["honest"].get(stat)) for q in PARENTS) + " |")
        print("\nPearson vs strong-tier difficulty\n| covariate | " + " | ".join(PARENTS) + " | mean |\n|"
              + "---|" * 6)
        for n, e in ev["per_cov"].items():
            print(f"| {n} | " + " | ".join(_c(e["per_parent"][q]["strong"].get("pearson")) for q in PARENTS)
                  + f" | {e['mean_pearson_strong']:+.3f} |")
        print("\nmatharena text-bearing:", {n: _c(e["per_parent"]["matharena"].get("honest_text_bearing", {})
                                                  .get("pearson")) for n, e in ev["per_cov"].items()})
        print("\npaired vs frozen_nested:")
        for n, d in ev["paired_vs_frozen_nested"].items():
            print(f"  {n}: " + "  ".join(f"{q[:10]} {_c(v)}" for q, v in d.items()))
        inn = ev["inner"]
        print("\ninner: mean gain in r_v by epoch", inn["mean_gain_r_v_by_epoch"], "runs improving",
              inn["runs_improving_v_by_epoch"], "best epoch counts", inn["best_epoch_counts"])
        print("idv r by epoch:", json.dumps(inn["mean_idv_r_by_epoch"]))
        for n, r in inn["runs"].items():
            print(f"  {n}: best {r['best_epoch']} alpha {r['ridge_alpha']} r_v "
                  + " ".join(f"{x:+.3f}" for x in r["r_v"]) + " | pair acc "
                  + " ".join("" if x is None else f"{x:.3f}" for x in r["train_pair_acc"])
                  + f" | {r['wall_s']:.0f}s")
    hz = s.get("harness")
    if hz:
        print("\n| covariate, line | test-like ± cluster SE (sel) | benchmark-equal | " + " | ".join(PARENTS)
              + " | mix/whole | R1 b / p | folds on | gate | placebo test-like |\n|" + "---|" * 12)
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
                      + f" | {ln['gate_pass']} | " + ("" if not pl else f"{pl.get('tl', float('nan')):+.5f}") + " |")
            print(f"  {n}: within-pair r {v.get('r_within_pair_tl')}, coverage {v['coverage_eval_items']}")
    if s.get("verdict"):
        print("\nverdict:", json.dumps(s["verdict"], indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=("data", "lora-cache", "lora", "eval", "harness", "verdict", "show"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if another LLM process is alive")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--runs", nargs="+", default=None, help="q:v run names (default: all 12)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="a smoke test: stop training after this many steps")
    ap.add_argument("--rows", default=H.ROWS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boots", type=int, default=BOOT)
    ap.add_argument("--placebo", type=int, default=N_PLACEBO)
    ap.add_argument("--covs", nargs="+", default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    {"data": stage_data, "lora-cache": stage_cache, "lora": stage_lora, "eval": stage_eval,
     "harness": stage_harness, "verdict": stage_verdict, "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
