"""Attempting instead of judging: Qwen3-4B-Instruct-2507 as a SUBJECT on matharena.

The 4B judge (paiec.llmfeat's digit rating) reads a task and guesses how likely
it is to be solved; on these items its rating moves the wrong way (see
references_vs_honest in the result). Here the same model attempts each problem k
times and the item is described by what the attempts look like, with no label
and no reference answer:

  top_share     share of the k attempts that give the modal answer (agreement)
  ans_entropy   entropy of the k answers, each unparsed attempt its own answer
  lp_answer     mean over attempts of the mean token log-prob of the answer
                inside \\boxed{} (raw model distribution, temperature 1)
  lp_answer_top the same over the attempts that give the modal answer only
  tok_lp        mean token log-prob over everything generated
  tok_entropy   mean next-token entropy over everything generated (nats)
  ent0          next-token entropy at the first generated token (D1: the answer's first token)
  fail_rate     no parseable \\boxed{} answer, or a refusal phrase
  trunc_rate    the attempt hit the token cap (D2)
  mean_len      generated tokens per attempt (D2; capped attempts count at the cap)

and, only to diagnose FLOOR, `graded`: the share of attempts whose answer equals
matharena's reference answer after normalisation (the platform input carries no
reference answer, so this is never a run-time feature).

Designs (DESIGNS): D1 answer only (k=8, at most 48 new tokens, the assistant
turn prefilled with "The final answer is $\\boxed{" so the first generated token
is the answer's, stopped at the closing brace); D2 short chain of thought (k=4,
at most D2_CAP new tokens, asked for brevity; an attempt that ends without a
\\boxed{} answer gets one forced continuation "...Final Answer: $\\boxed{" of at
most FORCE_TOKENS tokens, s1-style budget forcing). Sampling is the model card's
(temperature 0.7, top-p 0.8, top-k 20). Several items share one left-padded
batch; each prompt is prefilled once and its cache repeated k times (generate).
Batches are cut to a prefill and a cache budget (plan_batches).

Precision. The model runs in weight-only int8 (every linear layer and the tied
embedding, per-row absmax round-to-nearest, torch's MPS int8 matmul; 4.0 GB):
on the 16 GB M1 beside the desktop the fp16 model (8.05 GB) does not stay
resident and decodes at ~13 tokens a second. On 16 D1 items run both ways the
first-token entropy correlates 0.99 between fp16 and int8 (mean |difference|
0.05 nats) and the modal answer is the same on 15 (--precision fp16 exists for a
larger machine). Decoding is launch-bound, ~0.4-0.5 s a step at 8-16 rows on an
idle machine (up to ~1.2 s beside other jobs), and prefill runs at ~300 tokens a
second.

Items (select_items): the 160 matharena probe items of the attempt-signal
design: text only (no "See image"), a short checkable reference answer (integer
or expression; proofs and Kangaroo's image-only multiple choice excluded), >= 10
subjects, competitions with >= 10 such items, 10 per competition spread over
difficulty (every k-th item in b order). --items all takes every item that
passes the filters (the covariate run if the probe says GO). --per-comp N runs a
stratified subset (stratified(): N per competition, evenly spaced in difficulty,
round-robin order, so any prefix is spread). D2 ran on --per-comp 2 (32 items:
the easiest and the hardest probe item of each competition) with a 512-token
cap: decoding 16 rows took ~1.1 s a step on the shared machine, so the planned
D2 (160 items, 1,024 tokens) would take about 8 hours. D2 runs competition-major
(both of a competition's items together), so a stopped run holds whole
competitions.

Targets (stage target): the harness's honest difficulty, i.e. Rasch b
(testlike.rasch on official.eligible pairs) fitted without each of the five
subject folds of experiments/harness.py (fold_of), averaged over the folds
(what docs/findings.md "Item covariates with a known sign" uses for its sign
check); the in-sample b; and difficulty for the strong tier (Rasch b over the
subjects whose ability is at or above the median, a2/a7's tiers). References on
the same items: the existing 4B digit rating (higher = easier), the weak tier's
mean success, and Qwen3-4B-2507-Think's own success (a matharena subject: the
same 4B family attempting with full thinking).

Statistic: within-competition Spearman (ranks within competition, demeaned,
pooled Pearson), every feature oriented so that + means harder; 95% interval
from a bootstrap over competitions (2000 resamples), and per contest year (the
2026 contests are after the 4B's release, July 2025). Decision (the plan's
step 6): GO if the best label-free feature reaches rho >= 0.35 with the right
sign, CI lower bound > 0.15, and rho >= 0.25 on the 2026 contests; KILL if
every feature is below 0.15 in both designs; FLOOR if graded 4B accuracy is
under 10% (a stronger proxy would be needed).

Stages
  target    difficulty targets and the item list  -> data/attempt_probe/targets.parquet  (CPU, 1 min)
  gen       --design D1|D2 [--items probe|all] [--limit N] [--ipb N]: k attempts per item,
            resumable, one JSON line per item -> data/attempt_probe/<design>_<items>.jsonl.
            Prints the measured throughput per batch. Refuses to start if another
            language-model process is alive (MEMORY: 4 GB int8 weights plus up to KV_BUDGET
            of cache; the MPS pool is capped at ~8.3 GB so it fails rather than swaps).
  analyse   features vs targets, bootstrap, verdict -> results/attempt_probe.json
  covariate --design D --feature F: item_key -> value JSON for harness.py --stage eval,
            oriented + = harder, standardised within the benchmark and clipped at 3 sd

Run (repo root):
  python experiments/attempt_probe.py --stage target
  python experiments/attempt_probe.py --stage gen --design D1 --ipb 2          # ~8 min
  python experiments/attempt_probe.py --stage gen --design D2 --per-comp 2     # ~1.7 h (shared M1)
  python experiments/attempt_probe.py --stage analyse
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

# cap the MPS pool (a base of ~11.8 GB x ratio = 9.5 GB on the 16 GB M1, where ~8 GB is
# free beside the desktop): past it an allocation fails instead of swapping the machine
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTDIR = os.path.join(ROOT, "data", "attempt_probe")
RESULT = os.path.join(ROOT, "results", "attempt_probe.json")
REPO = "Qwen/Qwen3-4B-Instruct-2507"
BENCH = "matharena"
K_PER_COMP = 10
MIN_SUBJECTS = 10
D2_CAP = 512
FORCE_TOKENS = 32
FORCE_ROWS = 4                  # forced continuations per batch (their prompts hold a whole attempt)
KV_BYTES = 2 * 36 * 8 * 128 * 2  # cache bytes per token: 2 x layers x kv heads x head dim x fp16
PREFILL_TOKENS = 2400           # items x longest prompt per prefill forward
KV_BUDGET = 2.5e9               # rows x (prompt + new tokens) x KV_BYTES; weights 4.0 GB, and the
                                # un-grouped attention's per-layer key/value copies come on top
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)
SYS_D1 = "Answer with the final answer only. Do not show any working."
SYS_D2 = ("Solve the problem. Be concise: keep your reasoning short (under about 350 words), "
          "then give the final answer in \\boxed{}.")
PREFILL_D1 = "The final answer is $\\boxed{"
FORCE = "\n\n**Final Answer**\n\n$\\boxed{"
DESIGNS = {
    "D1": dict(k=8, max_new=48, ipb=2, system=SYS_D1, prefill=PREFILL_D1, force=False, cache="shared"),
    "D2": dict(k=4, max_new=D2_CAP, ipb=4, system=SYS_D2, prefill="", force=True, cache="static"),
}
YEAR_2026 = ("aime_2026", "aime_2026_I", "hmmt_feb_2026", "arxivmath_0126", "arxivmath_0226")
REFUSE = re.compile(r"cannot (be )?(determined|determine|solve)|not enough information|"
                    r"I'm unable|I am unable|insufficient information", re.I)
LLM_PROCS = "probe_attempts|llm_|lora|icl_probe|probe_compare|hidden_state|attempt_probe.py --stage gen"
BOOTS = 2000
#: feature -> sign so that sign * value is + for harder
FEATURES = {"top_share": -1, "ans_entropy": 1, "lp_answer": -1, "lp_answer_top": -1,
            "tok_lp": -1, "tok_entropy": 1, "ent0": 1, "fail_rate": 1, "trunc_rate": 1,
            "mean_len": 1}
PRIMARY = "top_share"          # declared before the run: self-consistency
#: ent0 in D2 is the entropy of a chain of thought's first token, < 1e-4 on most items
#: ("We are given ..."): not a feature there; D1 never reaches the cap
SKIP = {"D1": {"trunc_rate"}, "D2": {"ent0"}}


# --- answers ------------------------------------------------------------------------

def boxed(text, start=None):
    """Content of the last \\boxed{...} (or of the brace opened at `start`),
    None if absent or unclosed; (content, char span)."""
    if start is None:
        i = text.rfind("\\boxed{")
        if i < 0:
            return None, None
        start = i + 7
    j, depth = start, 1
    while j < len(text):
        c = text[j]
        depth += (c == "{") - (c == "}")
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
    s = (s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
         .replace("\\ln", "log").replace("\\log", "log").replace("\\infty", "oo")
         .replace("^", "**").replace("{", "(").replace("}", ")"))
    return s


def canon(a):
    """A normal form for comparing answers: numbers by value (6 significant
    digits), anything else as a cleaned string or sympy's expanded form."""
    if a is None:
        return None
    s = str(a).strip().strip("$").strip()
    s = _TEXT.sub(r"\1", s)
    for t in ("\\,", "\\;", "\\!", "\\ ", "~", " ", "\\%", "%", "^{\\circ}", "^\\circ", "\\circ"):
        s = s.replace(t, "")
    s = s.rstrip(".").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    if "=" in s:                        # "x=5", "21+49=70": the rightmost side
        s = s.rsplit("=", 1)[1]
    if not s:
        return None
    if re.fullmatch(r"[-+]?\d+", s):
        return "num:%.6g" % float(int(s)) if len(s) < 16 else "int:" + s.lstrip("+").lstrip("0")
    py = _latex_to_py(s)
    # parse_expr evaluates its input: only arithmetic over known names and one-letter symbols
    if (len(py) > 160 or not re.fullmatch(r"[0-9A-Za-z+\-*/().,]*", py)
            or py.count("**") > 2 or re.search(r"\*\*\(*\d{4,}", py)
            or re.search(r"factorial\(\d{5,}", py)
            or any(w not in _SAFE and not re.fullmatch(r"[A-Za-z]\d*", w)
                   for w in re.findall(r"[A-Za-z_]\w*", py))):
        return "str:" + s
    import signal

    def _timeout(*_):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, 2.0)         # 10^(10^10) and the like
    try:
        import sympy
        from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                                parse_expr, standard_transformations)
        e = parse_expr(py, evaluate=True,
                       transformations=standard_transformations + (implicit_multiplication_application,
                                                                   convert_xor))
        if not e.free_symbols:
            v = complex(e.evalf(15))
            if abs(v.imag) < 1e-12 and math.isfinite(v.real):
                return "num:%.6g" % v.real
        return "sym:" + str(sympy.expand(e))
    except BaseException as exc:          # parse errors, overflow, the alarm
        if isinstance(exc, KeyboardInterrupt):
            raise
        return "str:" + s
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# --- targets --------------------------------------------------------------------------

def _pairs():
    from paiec import data as D
    from paiec import official as O
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return O.eligible(D.load_pairs([BENCH]))


def select_items():
    """The probe selection (attempt-signal design a7), re-derived from the data."""
    from paiec.rasch import rasch
    it = pd.read_parquet(os.path.join(ROOT, "data", BENCH, "items.parquet")).set_index("item_id")
    r = pd.read_parquet(os.path.join(ROOT, "data", BENCH, "response.parquet"))
    r = r[r.response.isin([0.0, 1.0])]
    subs = {s: i for i, s in enumerate(r.subject_id.unique())}
    items = {j: i for i, j in enumerate(r.item_id.unique())}
    _, b = rasch(r.subject_id.map(subs).values, r.item_id.map(items).values,
                 r.response.values.astype(float), len(subs), len(items))
    df = pd.DataFrame({"b_sel": pd.Series(b, index=list(items)),
                       "n_subjects": r.groupby("item_id").subject_id.nunique()})
    df["comp"] = it.loc[df.index, "item_features"].str.extract(r"competition=([^;]+)")[0].values
    ref = it.loc[df.index, "grading_criterion"].map(lambda s: json.loads(s).get("reference_answer"))
    df["gold"] = ref.values

    def fmt(a):
        if a is None:
            return "none"
        a = str(a).strip()
        return "mcq" if re.fullmatch(r"[A-E]", a) else "integer" if re.fullmatch(r"-?\d+", a) else "expression"

    df["fmt"] = [fmt(a) for a in ref]
    df["image"] = it.loc[df.index, "content"].str.contains("See image").values
    pool = df[(~df.image) & df.fmt.isin(["integer", "expression"]) & (df.n_subjects >= MIN_SUBJECTS)]
    cc = pool.comp.value_counts()
    pool = pool[pool.comp.isin(cc[cc >= K_PER_COMP].index)]
    picked = []
    for _, g in pool.groupby("comp"):
        g = g.sort_values("b_sel")
        idx = np.linspace(0, len(g) - 1, min(K_PER_COMP, len(g))).round().astype(int)
        picked.append(g.iloc[np.unique(idx)])
    sel = pd.concat(picked)
    pool = pool.copy()
    pool["probe"] = pool.index.isin(sel.index)
    return pool


def stage_target():
    from paiec import testlike as T
    from paiec.rasch import rasch
    from experiments.harness import HONEST_FOLDS, fold_of
    pool = select_items()
    ps = _pairs()
    ins = T.rasch(ps)
    folds = [T.rasch([p for p in ps if fold_of(p.subject_id) != f]) for f in range(HONEST_FOLDS)]
    # tiers: Rasch ability over the eligible pairs' responses
    si, ii, y, keys = [], [], [], {}
    for j, p in enumerate(ps):
        for x in p.responses:
            si.append(j)
            ii.append(keys.setdefault(x.item_key, len(keys)))
            y.append(x.label)
    th, _ = rasch(np.array(si), np.array(ii), np.array(y, float), len(ps), len(keys))
    strong = [p for p, t in zip(ps, th) if t >= np.median(th)]
    weak = [p for p, t in zip(ps, th) if t < np.median(th)]
    b_strong = T.rasch(strong)

    def success(group):
        acc = {}
        for p in group:
            for x in p.responses:
                acc.setdefault(x.item_key, []).append(x.label)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    subj = pd.read_parquet(os.path.join(ROOT, "data", BENCH, "subjects.parquet"))
    q4_ids = set(subj.loc[subj.display_name == "Qwen3-4B-2507-Think", "subject_id"])
    q4 = [p for p in ps if p.subject_id in q4_ids]
    tgt = pool.copy()
    keys_ = tgt.index.astype(str)
    tgt["b_honest"] = [np.nanmean([f.get(k, np.nan) for f in folds]) if any(k in f for f in folds)
                       else np.nan for k in keys_]
    tgt["b_insample"] = [ins.get(k, np.nan) for k in keys_]
    tgt["b_strong"] = [b_strong.get(k, np.nan) for k in keys_]
    sw, ss = success(weak), success(strong)
    tgt["weak_success"] = [sw.get(k, np.nan) for k in keys_]
    tgt["strong_success"] = [ss.get(k, np.nan) for k in keys_]
    sq = success(q4)
    tgt["qwen4b_think_success"] = [sq.get(k, np.nan) for k in keys_]
    # the existing 4B digit rating (higher = easier), where extracted
    try:
        from paiec import llmfeat
        index, _, _, sc = llmfeat.load(os.path.join(ROOT, "data", "features"), BENCH)
        row = {iid: i for i, ids in enumerate(index["item_ids"]) for iid in ids}
        rat = sc["rating"].to_numpy()
        tgt["rating_4b"] = [rat[row[k]] if k in row else np.nan for k in keys_]
    except Exception as e:  # features absent: the reference is simply missing
        print("no 4B rating:", e)
        tgt["rating_4b"] = np.nan
    os.makedirs(OUTDIR, exist_ok=True)
    tgt.to_parquet(os.path.join(OUTDIR, "targets.parquet"))
    info = dict(n_pool=int(len(tgt)), n_probe=int(tgt.probe.sum()), comps=int(tgt[tgt.probe].comp.nunique()),
                n_pairs=len(ps), n_strong=len(strong), qwen4b_think_pairs=len(q4),
                corr_honest_insample=float(tgt[["b_honest", "b_insample"]].corr().iloc[0, 1]))
    print(json.dumps(info, indent=1))
    return tgt


# --- generation ---------------------------------------------------------------------

def other_llm_alive():
    """Processes other than this one (and its ancestors) whose command line names
    a language-model job; shell wrappers are skipped (their python child shows)."""
    out = subprocess.run(["ps", "-ax", "-o", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    procs = {}
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit():
            procs[int(parts[0])] = (int(parts[1]), parts[2])
    mine, pid = set(), os.getpid()
    while pid in procs and pid not in mine:
        mine.add(pid)
        pid = procs[pid][0]
    pat = re.compile(LLM_PROCS)
    return [f"{p} {c[:120]}" for p, (_, c) in procs.items()
            if p not in mine and pat.search(c) and not re.match(r"(\S*/)?(ba|z|da)?sh\b", c)
            and "pgrep" not in c and "grep " not in c]


class Recorder:
    """Per generated token: its log-prob and the next-token entropy under the
    raw model distribution (it runs before temperature / top-k / top-p)."""

    def __init__(self):
        self.prev, self.lp, self.ent = None, [], []

    def __call__(self, input_ids, scores):
        import torch
        ls = torch.log_softmax(scores.float(), -1)
        if self.prev is not None:
            self.lp.append(self.prev.gather(1, input_ids[:, -1:])[:, 0])
        self.ent.append(-(ls.exp() * ls).sum(-1))
        self.prev = ls
        return scores

    def finish(self, sequences):
        import torch
        self.lp.append(self.prev.gather(1, sequences[:, -1:])[:, 0])
        lp = torch.stack(self.lp, 1).cpu().numpy()
        ent = torch.stack(self.ent, 1).cpu().numpy()
        self.prev = None
        return lp, ent


class BraceStop:
    """D1: a sequence is done once the brace the prefill opened is closed."""

    def __init__(self, tok, plen):
        self.tok, self.plen = tok, plen

    def __call__(self, input_ids, scores, **kw):
        import torch
        done = []
        for row in input_ids[:, self.plen:].tolist():
            txt = self.tok.decode(row, skip_special_tokens=True)
            done.append(boxed("\\boxed{" + txt, start=7)[0] is not None)
        return torch.tensor(done, device=input_ids.device)


def _unmask_padded_rows():
    """Left-padded batches: a pad token's query row is fully masked, torch 2.6's
    MPS scaled_dot_product_attention returns NaN there, the NaN reaches the pad
    positions' keys and values in the next layer, and 0 * NaN poisons every real
    token (sampling then fails on NaN probabilities). transformers only repairs
    such rows for torch < 2.5. Let a fully masked row attend everywhere instead:
    its output is finite garbage that no real token ever reads."""
    import torch
    from transformers import masking_utils as MU
    orig = MU.sdpa_mask_recent_torch

    def sdpa_mask(*a, **kw):
        m = orig(*a, **kw)
        if m is not None and m.dtype == torch.bool:
            m = m | ~m.any(-1, keepdim=True)
        return m

    MU.AttentionMaskInterface._global_mapping["sdpa"] = sdpa_mask


def load_model(precision="int8"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from experiments.llm_features import snapshot
    _unmask_padded_rows()
    path, rev = snapshot(REPO)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if precision == "int8":
        model = _load_int8(path, "mps")
    else:
        model = _load_weights(AutoModelForCausalLM, path, torch.float16, "mps")
    model.generation_config = GenerationConfig.from_pretrained(path)     # both end tokens
    return model, tok, rev


def _qlinear_cls():
    import torch
    from torch import nn

    class QLinear(nn.Module):
        """Weight-only int8 linear layer: per-output-row absmax scales (round to
        nearest), torch's MPS int8 x fp16 kernel for decode-sized inputs and a
        dequantised fp16 matmul for prefill-sized ones."""

        def __init__(self, q, s):
            super().__init__()
            self.register_buffer("q", q)
            self.register_buffer("s", s)
            self.out_features, self.in_features = q.shape

        def forward(self, x):
            shp = x.shape
            x2 = x.reshape(-1, shp[-1]).contiguous()
            if x2.shape[0] <= 256:
                y = torch._weight_int8pack_mm(x2, self.q, self.s)
            else:
                y = x2 @ (self.q.to(x2.dtype) * self.s[:, None]).T
            return y.reshape(*shp[:-1], self.out_features)

    class QEmbedding(nn.Module):
        def __init__(self, q, s):
            super().__init__()
            self.register_buffer("q", q)
            self.register_buffer("s", s)

        def forward(self, ids):
            return self.q[ids].to(self.s.dtype) * self.s[ids][..., None]

    return QLinear, QEmbedding


def quantize_rows(w, dev, dtype, chunk=16384):
    """int8 round-to-nearest with one absmax scale per row, chunked on the CPU."""
    import torch
    qs, ss = [], []
    for i in range(0, w.shape[0], chunk):
        c = w[i:i + chunk].float()
        if not torch.isfinite(c).all():
            raise RuntimeError("non-finite weight")
        sc = (c.abs().amax(1) / 127.0).clamp_min(1e-8)
        qs.append(torch.round(c / sc[:, None]).clamp(-127, 127).to(torch.int8))
        ss.append(sc)
    return torch.cat(qs).to(dev), torch.cat(ss).to(dtype).to(dev)


def _load_int8(path, dev):
    """Qwen3 with every linear layer and the (tied) embedding in weight-only int8,
    4.0 GB instead of 8.05 GB: on the 16 GB M1 beside the desktop the fp16 model
    does not stay resident and decodes at ~13 tokens a second (swapping)."""
    import glob
    import torch
    from torch import nn
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen3 import modeling_qwen3 as Q
    from experiments.llm_features import _no_sdpa_gqa
    QLinear, QEmbedding = _qlinear_cls()
    cfg = AutoConfig.from_pretrained(path)
    _no_sdpa_gqa()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16, attn_implementation="sdpa")
    handles = [safe_open(f, framework="pt", device="cpu") for f in sorted(glob.glob(os.path.join(path, "*.safetensors")))]
    where = {n: h for h in handles for n in h.keys()}

    def get(name):
        return where[name].get_tensor(name)

    with torch.no_grad():
        for mname, mod in list(model.named_modules()):
            for cname, child in list(mod.named_children()):
                full = f"{mname}.{cname}" if mname else cname
                if isinstance(child, nn.Linear) and full != "lm_head":
                    if child.bias is not None:
                        raise ValueError(f"{full} has a bias")
                    setattr(mod, cname, QLinear(*quantize_rows(get(full + ".weight"), dev, torch.float16)))
            torch.mps.empty_cache()
        qe, se = quantize_rows(get("model.embed_tokens.weight"), dev, torch.float16)
        model.model.embed_tokens = QEmbedding(qe, se)
        if not getattr(cfg, "tie_word_embeddings", False):
            raise ValueError("expected tied embeddings")
        model.lm_head = QLinear(qe, se)
        for mname, mod in list(model.named_modules()):
            for pname, prm in list(mod.named_parameters(recurse=False)):
                if prm.is_meta:
                    full = f"{mname}.{pname}" if mname else pname
                    t = get(full).to(torch.float16)
                    if not torch.isfinite(t).all():
                        raise RuntimeError(f"non-finite weight {full}")
                    setattr(mod, pname, nn.Parameter(t.to(dev), requires_grad=False))
        model.model.rotary_emb = Q.Qwen3RotaryEmbedding(config=cfg, device=dev)
    left = [n for n, b in list(model.named_parameters()) + list(model.named_buffers()) if b.is_meta]
    if left:
        raise RuntimeError(f"still on meta: {left[:5]}")
    torch.mps.empty_cache()
    return model.eval()


def _load_weights(model_cls, path, dtype, dev):
    """experiments/llm_features.load_weights with the finiteness check on the CPU
    and the MPS staging buffers released after every checkpoint file (it leaves
    ~1.4 GB cached and its on-device check of the 0.4 G-entry embedding needs
    another 0.7 GB, which the capped pool does not have)."""
    import glob
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights
    from experiments.llm_features import _no_sdpa_gqa
    cfg = AutoConfig.from_pretrained(path)
    _no_sdpa_gqa()
    with no_init_weights(), torch.device(dev):
        model = model_cls.from_config(cfg, dtype=dtype, attn_implementation="sdpa")
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded = set()
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, framework="pt", device="cpu") as fh:
                for name in fh.keys():
                    target = name if name in params else ("model." + name if "model." + name in params else None)
                    if target is None:
                        if name == "lm_head.weight":
                            continue
                        raise KeyError(f"{name} of {f} has no place in the model")
                    t = fh.get_tensor(name).to(dtype)
                    if not torch.isfinite(t).all():
                        raise RuntimeError(f"non-finite weight {name} in {dtype}")
                    params[target].copy_(t)
                    loaded.add(target)
                    del t
            torch.mps.synchronize()
            torch.mps.empty_cache()
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
        loaded.update(n for n in params if n.endswith("lm_head.weight"))
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"weights missing from the checkpoint: {missing[:5]}")
    return model.eval()


def prompt_ids(tok, content, d):
    msgs = [{"role": "system", "content": d["system"]}, {"role": "user", "content": content}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + d["prefill"]
    return tok(text, add_special_tokens=False)["input_ids"]


def token_span(tok, ids, c0, c1):
    """Token range [t0, t1) of `ids` whose decoded text covers chars [c0, c1)."""
    f = lambda t: len(tok.decode(ids[:t], skip_special_tokens=True))  # noqa: E731
    lo, hi = 0, len(ids)
    while lo < hi:                       # first t with f(t + 1) > c0
        m = (lo + hi) // 2
        if f(m + 1) > c0:
            hi = m
        else:
            lo = m + 1
    t0 = lo
    lo, hi = t0, len(ids)
    while lo < hi:                       # first t with f(t) >= c1
        m = (lo + hi) // 2
        if f(m) >= c1:
            hi = m
        else:
            lo = m + 1
    return t0, max(lo, t0 + 1)


def generate(model, tok, prompts, max_new, seed, stop=None, share=1, static=False):
    """Sample `share` continuations of each prompt (a left-padded batch of
    len(prompts) * share rows, row r*share+j is prompt r's j-th sample).

    The prompts are prefilled once, all but their last token, and the cache is
    repeated `share` times before sampling (generate() continues from a filled
    cache), so k attempts of an item cost one prefill. Position ids follow
    generate()'s own rule for padded rows (cumulative sum of the mask). Returns
    per row (generated ids up to the first end token, token log-probs,
    entropies, ended, capped).

    static=True instead expands the prompts first and lets generate() prefill
    them into a preallocated static cache: long generations with a growing
    dynamic cache fragment the MPS pool (9.7 GB held for 6.3 GB live after 160
    steps of 16 rows) until the capped pool runs out; the static one holds 7.7."""
    import torch
    from transformers import DynamicCache, LogitsProcessorList, StoppingCriteriaList
    L = max(len(p) for p in prompts)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(prompts), L), pad, dtype=torch.long)
    mask = torch.zeros((len(prompts), L), dtype=torch.long)
    for r, p in enumerate(prompts):
        ids[r, L - len(p):] = torch.tensor(p)
        mask[r, L - len(p):] = 1
    ids, mask = ids.to("mps"), mask.to("mps")
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    rec = Recorder()
    with torch.inference_mode():
        if static:
            ids, mask = ids.repeat_interleave(share, 0), mask.repeat_interleave(share, 0)
            torch.manual_seed(seed)
            out = model.generate(input_ids=ids, attention_mask=mask, do_sample=True, max_new_tokens=max_new,
                                 pad_token_id=pad, cache_implementation="static",
                                 logits_processor=LogitsProcessorList([rec]),
                                 stopping_criteria=StoppingCriteriaList([stop(tok, L)]) if stop else None,
                                 **SAMPLING)
            return _rows(out, rec, L, eos, max_new)
        cache = DynamicCache()
        if L > 1:
            pos = (mask[:, :-1].cumsum(-1) - 1).masked_fill(mask[:, :-1] == 0, 1)
            model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos, past_key_values=cache,
                  use_cache=True, logits_to_keep=1)
        if share > 1:
            cache.batch_repeat_interleave(share)
            ids, mask = ids.repeat_interleave(share, 0), mask.repeat_interleave(share, 0)
        torch.manual_seed(seed)
        out = model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, do_sample=True,
                             max_new_tokens=max_new, pad_token_id=pad,
                             logits_processor=LogitsProcessorList([rec]),
                             stopping_criteria=StoppingCriteriaList([stop(tok, L)]) if stop else None,
                             **SAMPLING)
    del cache
    return _rows(out, rec, L, eos, max_new)


def _rows(out, rec, L, eos, max_new):
    lp, ent = rec.finish(out)
    gen = out[:, L:].cpu().tolist()
    rows = []
    for r, g in enumerate(gen):
        n = next((t for t, x in enumerate(g) if x in eos), None)
        ended = n is not None
        n = len(g) if n is None else n
        rows.append((g[:n], lp[r, :n], ent[r, :n], ended, n >= max_new))
    return rows


def plan_batches(order, lens, d):
    """Consecutive items per batch, at most d['ipb'], within a prefill budget
    (items x longest prompt) and a cache budget (rows x (prompt + new tokens))."""
    out, cur = [], []
    for iid in order:
        cand = cur + [iid]
        L = max(lens[i] for i in cand)
        rows_prefilled = len(cand) * (d["k"] if d["cache"] == "static" else 1)   # static: no sharing
        too_big = (len(cand) > d["ipb"] or rows_prefilled * L > PREFILL_TOKENS * (2 if d["cache"] == "static" else 1)
                   or len(cand) * d["k"] * (L + d["max_new"]) * KV_BYTES > KV_BUDGET)
        if cur and too_big:
            out.append(cur)
            cur = [iid]
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def attempt_batch(model, tok, contents, d, seed):
    """k attempts for each item in `contents`; per item a list of sample dicts."""
    import torch
    k = d["k"]
    base = [prompt_ids(tok, c, d) for c in contents]
    prompts = [p for p in base for _ in range(k)]
    t0 = time.time()
    rows = generate(model, tok, base, d["max_new"], seed, stop=BraceStop if d["prefill"] else None, share=k,
                    static=d["cache"] == "static")
    secs = time.time() - t0
    samples = []
    for (g, lp, ent, ended, capped) in rows:
        txt = tok.decode(g, skip_special_tokens=True)
        full = d["prefill"] + txt
        if d["prefill"]:
            ans, span = boxed(full, start=len(d["prefill"]))
        else:
            ans, span = boxed(full)
        smp = dict(text=txt, n_tok=len(g), capped=bool(capped), ended=bool(ended), forced=False,
                   tok_lp=float(np.mean(lp)) if len(lp) else float("nan"),
                   tok_entropy=float(np.mean(ent)) if len(ent) else float("nan"),
                   ent0=float(ent[0]) if len(ent) else float("nan"),
                   refuse=bool(REFUSE.search(txt)))
        if ans is not None:
            c0, c1 = span[0] - len(d["prefill"]), span[1] - len(d["prefill"])
            t_0, t_1 = token_span(tok, g, max(c0, 0), max(c1, 1))
            smp["lp_answer"] = float(np.mean(lp[t_0:t_1]))
        smp["ans"] = ans
        samples.append(smp)
    gen_tok = sum(len(r[0]) for r in rows)
    # budget forcing: one short continuation for attempts that gave no answer
    if d["force"]:
        need = [i for i, s in enumerate(samples) if s["ans"] is None]
        if hasattr(model, "_cache"):                  # the static cache of the main pass
            del model._cache                          # (generate() re-creates it when absent)
            torch.mps.empty_cache()
        if need:
            fp = [prompts[i] + list(rows[i][0]) + tok(FORCE, add_special_tokens=False)["input_ids"]
                  for i in need]
            t1 = time.time()
            frows = []
            for j in range(0, len(fp), FORCE_ROWS):
                try:
                    frows += generate(model, tok, fp[j:j + FORCE_ROWS], FORCE_TOKENS, seed + 7919 + j,
                                      stop=BraceStop)
                except RuntimeError as e:          # out of memory: those attempts stay unanswered
                    print("forced continuation failed:", str(e)[:120], flush=True)
                    frows += [None] * len(fp[j:j + FORCE_ROWS])
                torch.mps.empty_cache()
            secs += time.time() - t1
            for i, fr in zip(need, frows):
                if fr is None:
                    continue
                g, lp, ent, ended, capped = fr
                txt = tok.decode(g, skip_special_tokens=True)
                ans, span = boxed("\\boxed{" + txt, start=7)
                s = samples[i]
                s.update(forced=True, forced_text=txt, ans=ans)
                if ans is not None:
                    t_0, t_1 = token_span(tok, g, 0, max(span[1] - 7, 1))
                    s["lp_answer"] = float(np.mean(lp[t_0:t_1]))
                gen_tok += len(g)
    for s in samples:
        s["canon"] = canon(s["ans"])
    per_item = [samples[i * k:(i + 1) * k] for i in range(len(contents))]
    return per_item, secs, gen_tok, sum(len(p) for p in base)


def stratified(todo, per_comp, comp_major=False):
    """per_comp items of each competition, evenly spaced in difficulty rank (the
    selection's b), ordered round-robin over competitions, extremes first (rank
    positions 0, N-1, 1, N-2, ...), so that any prefix of whole rounds is spread
    over competitions and over difficulty within each."""
    rows = []
    for comp, g in todo.groupby("comp"):
        g = g.sort_values("b_sel")
        idx = np.unique(np.linspace(0, len(g) - 1, min(per_comp, len(g))).round().astype(int))
        n = len(idx)
        rnd = [x for pair in zip(range(n), range(n - 1, -1, -1)) for x in pair]
        first = list(dict.fromkeys(rnd))[:n]              # 0, n-1, 1, n-2, ...
        rows += [(first.index(j), comp, iid) for j, iid in enumerate(g.index[idx])]
    rr = [iid for _, _, iid in sorted(rows)]
    if not comp_major:
        return rr
    # competition-major: each competition's items together (extremes first), the
    # competitions in the order they first appear in the round-robin order, so any
    # prefix holds whole competitions (a within-competition rank needs both ends)
    comp_of = todo.comp.to_dict()
    first_seen = {c: n for n, c in reversed(list(enumerate(comp_of[i] for i in rr)))}
    return sorted(rr, key=lambda i: (first_seen[comp_of[i]], rr.index(i)))


def stage_gen(design, items_kind, limit, ipb, precision="int8", per_comp=0):
    import torch
    alive = other_llm_alive()
    if alive:
        raise SystemExit("another language-model process is alive, not loading the 4B:\n" + "\n".join(alive))
    d = dict(DESIGNS[design])
    if ipb:
        d["ipb"] = ipb
    tgt = pd.read_parquet(os.path.join(OUTDIR, "targets.parquet"))
    todo = tgt[tgt.probe] if items_kind == "probe" else tgt
    it = pd.read_parquet(os.path.join(ROOT, "data", BENCH, "items.parquet")).set_index("item_id")
    path = os.path.join(OUTDIR, f"{design}_{items_kind}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(ln)["item_id"] for ln in fh if ln.strip()}
    order = [i for i in (stratified(todo, per_comp, comp_major=design == "D2") if per_comp else todo.index)
             if i not in done]
    if limit:
        order = order[:limit]
    print(f"{design} {items_kind}: {len(done)} done, {len(order)} to go, {d['ipb']} items x k={d['k']} per batch",
          flush=True)
    if not order:
        return
    model, tok, rev = load_model(precision)
    print(f"model {REPO}@{rev} on mps, {torch.mps.current_allocated_memory() / 1e9:.2f} GB allocated", flush=True)
    lens = {i: len(prompt_ids(tok, it.loc[i, "content"], d)) for i in order}
    batches = plan_batches(order, lens, d)
    print(f"{len(batches)} batches", flush=True)
    t_all, tok_all, b0 = time.time(), 0, 0
    with open(path, "a") as fh:
        for ids in batches:
            seed = int(todo.index.get_loc(ids[0]))
            per_item, secs, gen_tok, pre_tok = attempt_batch(model, tok, [it.loc[i, "content"] for i in ids], d,
                                                             seed)
            tok_all += gen_tok
            for iid, smp in zip(ids, per_item):
                fh.write(json.dumps(dict(item_id=iid, design=design, rev=rev, precision=precision, seed=seed,
                                         samples=smp)) + "\n")
            fh.flush()
            mem = torch.mps.driver_allocated_memory() / 1e9
            el = time.time() - t_all
            left = (len(order) - b0 - len(ids)) / max(1, b0 + len(ids)) * el
            print(f"[{b0 + len(ids)}/{len(order)}] {secs:.1f}s {gen_tok} gen tok ({gen_tok / secs:.0f} tok/s agg), "
                  f"{pre_tok} prompt tok, mps {mem:.1f} GB, eta {left / 60:.0f} min", flush=True)
            b0 += len(ids)
            torch.mps.empty_cache()
    print(f"done: {tok_all} tokens in {(time.time() - t_all) / 60:.1f} min", flush=True)


# --- analysis --------------------------------------------------------------------------

def item_features(samples, gold):
    k = len(samples)
    can = [canon(s["ans"]) for s in samples]        # recomputed: canon() may have been refined
    parsed = [c for c in can if c is not None]
    cnt = Counter(parsed)
    top, topn = cnt.most_common(1)[0] if cnt else (None, 0)
    cats = list(cnt.values()) + [1] * (k - len(parsed))     # each unparsed attempt its own answer
    p = np.array(cats, float) / k
    lpa = [s["lp_answer"] for s in samples if "lp_answer" in s]
    lpt = [s["lp_answer"] for s, c in zip(samples, can) if "lp_answer" in s and c == top]
    g = canon(gold)
    return dict(
        top_share=topn / k, ans_entropy=float(-(p * np.log(p)).sum()),
        lp_answer=float(np.mean(lpa)) if lpa else np.nan,
        lp_answer_top=float(np.mean(lpt)) if lpt else np.nan,
        tok_lp=float(np.nanmean([s["tok_lp"] for s in samples])),
        tok_entropy=float(np.nanmean([s["tok_entropy"] for s in samples])),
        ent0=float(np.nanmean([s["ent0"] for s in samples])),
        fail_rate=float(np.mean([(c is None) or s["refuse"] for s, c in zip(samples, can)])),
        trunc_rate=float(np.mean([s["capped"] for s in samples])),
        forced_rate=float(np.mean([s["forced"] for s in samples])),
        mean_len=float(np.mean([s["n_tok"] for s in samples])),
        graded=float(np.mean([c is not None and c == g for c in can])),
        top_correct=float(top is not None and top == g),
        n_distinct=len(cnt))


def load_design(design, items_kind="probe"):
    path = os.path.join(OUTDIR, f"{design}_{items_kind}.jsonl")
    if not os.path.exists(path):
        return None
    tgt = pd.read_parquet(os.path.join(OUTDIR, "targets.parquet"))
    rows = {}
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                o = json.loads(ln)
                rows[o["item_id"]] = item_features(o["samples"], tgt.loc[o["item_id"], "gold"])
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.join(tgt, how="left")


def _within_ranks(x, g):
    d = pd.DataFrame({"x": x, "g": g})
    return d.groupby("g").x.transform(lambda s: s.rank(pct=True) - s.rank(pct=True).mean()).to_numpy()


def within_rho(x, y, g, rng, boots=BOOTS):
    """Within-group Spearman (group-demeaned percentile ranks, pooled Pearson),
    with a bootstrap over groups; groups with one distinct value add nothing."""
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float), "g": np.asarray(g)}).dropna()
    d = d[d.groupby("g").x.transform("size") >= 2]        # a lone item has no within-group rank
    d["x"] = d.x.round(3)       # differences below 1e-3 (a CoT's first-token entropy is ~0) are ties
    if len(d) < 8 or d.x.nunique() < 2:
        return None
    xr, yr, gs = _within_ranks(d.x, d.g), _within_ranks(d.y, d.g), d.g.to_numpy()
    if xr.std() == 0:
        return None
    rho = float(np.corrcoef(xr, yr)[0, 1])
    ug = np.unique(gs)
    at = {u: np.flatnonzero(gs == u) for u in ug}
    bs = []
    for _ in range(boots):
        ii = np.concatenate([at[u] for u in rng.choice(ug, len(ug))])
        if xr[ii].std() > 0 and yr[ii].std() > 0:
            bs.append(np.corrcoef(xr[ii], yr[ii])[0, 1])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 50 else [None, None]
    return dict(n=int(len(d)), groups=int(len(ug)), rho=round(rho, 3),
                ci=[None if c is None else round(c, 3) for c in ci])


def verdict(rep):
    """The plan's decision rule on the analysed designs."""
    rep = {k: v for k, v in rep.items() if k in DESIGNS}
    acc = max(v["accuracy_4B"] for v in rep.values())
    best = None
    for dsg, v in rep.items():
        for f, row in v["features_vs_honest"].items():
            if row and (best is None or row["rho"] > best[2]["rho"]):
                best = (dsg, f, row)
    y26 = rep[best[0]]["by_year_honest"][best[1]]["2026"]
    all_below = all((row is None or row["rho"] < 0.15) for v in rep.values()
                    for row in v["features_vs_honest"].values())
    if acc < 0.10:
        call = "FLOOR"
    elif (best[2]["rho"] >= 0.35 and best[2]["ci"][0] is not None and best[2]["ci"][0] > 0.15
          and y26 is not None and y26["rho"] >= 0.25):
        call = "GO"
    elif all_below:
        call = "KILL"
    else:
        call = "WEAK (between KILL and GO)"
    return dict(call=call, best_design=best[0], best_feature=best[1], best=best[2], best_2026=y26,
                best_graded_accuracy=round(acc, 3))


def stage_analyse():
    rng = np.random.default_rng(0)
    tgt = pd.read_parquet(os.path.join(OUTDIR, "targets.parquet"))
    probe = tgt[tgt.probe]
    rep = {}
    refs = {}
    for name, col, sgn in (("rating_4b (judge, higher = easier)", "rating_4b", -1),
                           ("weak tier mean success", "weak_success", -1),
                           ("Qwen3-4B-2507-Think success (matharena subject)", "qwen4b_think_success", -1),
                           ("in-sample b", "b_insample", 1)):
        refs[name] = within_rho(sgn * probe[col], probe.b_honest, probe.comp, rng)
    refs["strong-tier b vs honest b"] = within_rho(probe.b_strong, probe.b_honest, probe.comp, rng)
    for design in DESIGNS:
        df = load_design(design)
        if df is None or not len(df):
            continue
        df["year"] = np.where(df.comp.isin(YEAR_2026), "2026", "2025")
        v = dict(n_items=int(len(df)), k=DESIGNS[design]["k"], max_new=DESIGNS[design]["max_new"],
                 accuracy_4B=round(float(df.graded.mean()), 3),
                 majority_accuracy_4B=round(float(df.top_correct.mean()), 3),
                 accuracy_by_year={y: round(float(g.graded.mean()), 3) for y, g in df.groupby("year")},
                 accuracy_by_fmt={f: round(float(g.graded.mean()), 3) for f, g in df.groupby("fmt")},
                 fail_rate=round(float(df.fail_rate.mean()), 3), trunc_rate=round(float(df.trunc_rate.mean()), 3),
                 forced_rate=round(float(df.forced_rate.mean()), 3), mean_len=round(float(df.mean_len.mean()), 1),
                 features_vs_honest={}, features_vs_strong={}, by_year_honest={})
        for f, sgn in FEATURES.items():
            if f not in df or f in SKIP[design] or df[f].nunique(dropna=True) < 2:
                continue
            v["features_vs_honest"][f] = within_rho(sgn * df[f], df.b_honest, df.comp, rng)
            v["features_vs_strong"][f] = within_rho(sgn * df[f], df.b_strong, df.comp, rng)
            v["by_year_honest"][f] = {y: within_rho(sgn * g[f], g.b_honest, g.comp, rng)
                                      for y, g in df.groupby("year")}
        v["graded_vs_honest (diagnostic, needs the reference answer)"] = within_rho(
            -df.graded, df.b_honest, df.comp, rng)
        v["graded_by_year"] = {y: within_rho(-g.graded, g.b_honest, g.comp, rng) for y, g in df.groupby("year")}
        v["rating_4b_same_items"] = within_rho(-df.rating_4b, df.b_honest, df.comp, rng)
        # partial: does agreement add to the judge? rank-residual of the feature on the rating
        multi = df[df.groupby("comp").comp.transform("size") >= 2]
        if len(multi) >= 8 and multi.rating_4b.notna().all():
            xr = _within_ranks(-multi.top_share, multi.comp)
            rr = _within_ranks(-multi.rating_4b, multi.comp)
            yr = _within_ranks(multi.b_honest, multi.comp)
            if rr.std() > 0 and xr.std() > 0:
                res = xr - (rr @ xr) / (rr @ rr) * rr
                v["top_share_partial_on_rating"] = round(float(np.corrcoef(res, yr)[0, 1]), 3)
        # duplicates (the same problem text in two competitions): feature test-retest
        it = pd.read_parquet(os.path.join(ROOT, "data", BENCH, "items.parquet")).set_index("item_id")
        body = it.loc[df.index, "content"].map(lambda s: s.split("\n\n", 1)[-1])
        dup = body[body.duplicated(keep=False)]
        pairs = [list(g.index) for _, g in dup.groupby(dup) if len(g) == 2]
        if len(pairs) >= 5:
            a = np.array([[df.loc[p[0], PRIMARY], df.loc[p[1], PRIMARY]] for p in pairs])
            v["duplicate_texts"] = dict(n_pairs=len(pairs),
                                        test_retest_r=round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3))
        # checks on the reasoning's token entropy: one attempt at a time, the prompt's
        # length as a rival, and (two items per competition) the concordant competitions
        raw = {}
        with open(os.path.join(OUTDIR, f"{design}_probe.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    o = json.loads(ln)
                    raw[o["item_id"]] = o["samples"]
        v["tok_entropy_single_attempt"] = [
            within_rho(pd.Series({i: raw[i][j]["tok_entropy"] for i in df.index}), df.b_honest, df.comp, rng)
            for j in range(DESIGNS[design]["k"])]
        items = pd.read_parquet(os.path.join(ROOT, "data", BENCH, "items.parquet")).set_index("item_id")
        plen = items.loc[df.index, "content"].str.len()
        v["prompt_chars_same_items"] = within_rho(plen, df.b_honest, df.comp, rng)
        v["tok_entropy_vs_prompt_chars"] = within_rho(df.tok_entropy, plen, df.comp, rng)
        sizes = df.groupby("comp").size()
        if (sizes == 2).all():
            conc = df.sort_values("b_honest").groupby("comp").tok_entropy.apply(
                lambda x: int(np.sign(x.iloc[-1] - x.iloc[0])))
            v["tok_entropy_pairs"] = dict(concordant=int((conc > 0).sum()), discordant=int((conc < 0).sum()),
                                          competitions=int(len(conc)))
        rep[design] = v
    # D1 on exactly D2's items (D2 ran on a stratified subset): paired comparison
    if "D1" in rep and "D2" in rep:
        d1, d2 = load_design("D1"), load_design("D2")
        common = d1.index.intersection(d2.index)
        sub = d1.loc[common]
        rep["D1_on_D2_items"] = {"n_items": int(len(common)), "features_vs_honest": {
            f: within_rho(sgn * sub[f], sub.b_honest, sub.comp, rng) for f, sgn in FEATURES.items()
            if f in sub and sub[f].nunique(dropna=True) > 1}}
    out = dict(meta=dict(model=REPO, sampling=SAMPLING, designs={k: {kk: vv for kk, vv in d.items()}
                                                                 for k, d in DESIGNS.items()},
                         primary_feature=PRIMARY, year_2026=list(YEAR_2026), boots=BOOTS,
                         target="honest: mean over the harness's 5 subject folds of testlike.rasch "
                                "(official.eligible matharena pairs) without that fold",
                         n_probe=int(len(probe)), comps=int(probe.comp.nunique())),
               references_vs_honest=refs, designs=rep)
    if rep:
        out["decision"] = verdict(rep)
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


def stage_covariate(design, feature, items_kind):
    df = load_design(design, items_kind)
    x = FEATURES[feature] * df[feature]
    z = (x - x.mean()) / x.std()
    path = os.path.join(OUTDIR, f"cov_{design}_{feature}_{items_kind}.json")
    with open(path, "w") as fh:
        json.dump({str(k): float(np.clip(v, -3, 3)) for k, v in z.items() if np.isfinite(v)}, fh)
    print(path, len(z))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["target", "gen", "analyse", "covariate"])
    ap.add_argument("--design", choices=list(DESIGNS))
    ap.add_argument("--items", default="probe", choices=["probe", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ipb", type=int, default=0, help="items per batch (default: the design's)")
    ap.add_argument("--feature", default=PRIMARY)
    ap.add_argument("--precision", default="int8", choices=["int8", "fp16"])
    ap.add_argument("--per-comp", type=int, default=0, help="stratified subset: items per competition")
    a = ap.parse_args()
    if a.stage == "target":
        stage_target()
    elif a.stage == "gen":
        stage_gen(a.design, a.items, a.limit, a.ipb, a.precision, a.per_comp)
    elif a.stage == "analyse":
        stage_analyse()
    else:
        stage_covariate(a.design, a.feature, a.items)
