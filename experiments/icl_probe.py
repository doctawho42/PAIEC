"""Few-shot prompting over a pair's own labels: does Qwen3-4B-Instruct-2507, shown
the items THIS subject solved and failed so far, order the subject's remaining
items better than without them?

The protocol allows few-shot evidence only through the pair's revealed labels
(at most 31). This probe is the minimal test of in-context learning on them, for
the report (docs/findings.md, "Few-shot prompting"); the plan's step 10.

Appearances (--stage select, CPU). Test-like pair appearances from the harness's
stored rows (experiments/harness.py, data/harness_rows/tl: the shipped hier
replayed on testlike.Regime() runs 0..299, seed 2), at budget B = 15: the pair's
first 15 acquired labels in the platform's order, and its evaluated items. An
appearance is eligible when (i) it has 15 own labels, 2 to 13 of them successes
(a prompt with one class only teaches nothing about items), (ii) at least 20
evaluated items, at least MIN_CLASS of them solved and MIN_CLASS failed (a
within-pair correlation with the outcome needs both), and (iii) at least
TEXT_SHARE of its items carry text (no image placeholder, itemcov.image_ref;
on matharena also >= 70 characters, llm4b_close.text_bearing, whose shorter
items are a system prompt or a date and an id; Kangaroo's image-only problems
carry nothing a reader could use). 1,236 of the 2,425 test-like appearances
pass all three:
the selection favours pairs whose rate is not near 0 or 1, where ordering
items matters most, so it is kind to the method. PER_PARENT appearances per
parent are taken in a seeded hash order with distinct subjects. Evaluated items
are subsampled to MAX_EVAL in a hash order. The plan stores, per appearance,
hier's B15 prediction for every evaluated item (ev_p[4]) and for every labeled
item as if unlabeled (acq_pu[4]), and the honest difficulty of every item: Rasch
b fitted on the parent's subjects outside the pair's subject fold
(harness.oracle_maps, fold harness.fold_of(subject)), so the subject's own
responses never enter it.

Prompts (--stage run, the language model). System SYSTEM; one user turn:
  icl   INTRO, the 15 labeled items as EXAMPLE blocks (each task cut to EX_TOKENS
        by llmfeat.head_tail with llmfeat.MARKER; outcome SOLVED / FAILED), in
        acquisition order, then BRIDGE and the target task;
  zs    the zero-shot control on the same template: BRIDGE_ZS and the target;
then SUFFIX, and the assistant header. The target task is cut to TGT_TOKENS.
The text of an item is llmfeat.item_text (item_content, a newline,
item_features). The score is log p(Yes) - log p(No) of the first assistant
token, one forward pass, nothing generated; the probability mass on the two
tokens is kept as a check. The prompt prefix (everything before the target) is
run once and its key/value cache reused for each target (DynamicCache.crop back
to the prefix), so a target sees exactly the prefix and itself.
Scored item sets (passes, run in this order so the kill rule's inputs finish
first):
  core  icl (15 examples) on the evaluated items; zs on the evaluated and the
        labeled items; the labeled items cross-fitted: the labels are split in
        two halves (alternating in a hash order) and each half is scored by the
        prompt whose examples are the other half (icl_cf), so no labeled item's
        score has seen its own label (the per-pair slope is fitted on them).
  perm  the 15 examples with their outcomes permuted (a seeded permutation:
        the base rate is kept, the item-outcome link is broken) on the evaluated
        items: separates learning from the labels from seeing the benchmark's
        items and base rate.
  half  each half-prompt (7 or 8 examples) on the evaluated items (icl_half, the
        mean of the two): a dose-response between 0, ~7.5 and 15 examples.
Raw scores go to data/icl_probe/scores/<appearance>_<pass>.json (atomic,
resumable). The run refuses to start while another language-model process is
alive (other_llm_processes).

The model is Qwen3-4B-Instruct-2507 from the local Hugging Face cache (no
download) in weight-only int8 (experiments/attempt_probe.load_model: every linear
layer and the tied embedding, per-row absmax round-to-nearest, 4.0 GB; the
fp16 model does not stay resident on the 16 GB M1 beside other work, and
attempt_probe measured int8 against fp16 at 0.98-0.99 correlation on log-probs
and entropies).

--stage analyse (CPU). Per appearance, over its evaluated items: Pearson r of
each score with the honest easiness -b (r_diff, the harness value map's index),
with the outcome K/N (r_out, point-biserial for single responses) and its AUC,
and r_out and r_diff net of hier's B15 logit (r_out_partial, r_diff_partial:
what the score adds to the model that ships). References on the same items: -b itself and hier's B15 logit.
Means over the appearances (equal weight; 4 per parent, so also benchmark-equal)
with the SE over appearances and the SE across the four parent means; paired
differences icl - zs, icl - perm, icl - half, half - zs.
Kill rule (declared in the plan before the run): KILL in-context learning if
r_diff(icl) - r_diff(zs) < KILL_GAIN or r_diff(icl) < KILL_R. r_out is reported
beside it.
Implied ALC: the mean within-pair r_diff read off the harness's honest gate
table (results/harness_thresholds.json, tables.honest: r_within_pair_tl against
the test-like ALC difference of each line), by linear interpolation; r <= 0
reads the r = 0 row. The table's covariates act at every budget from B1 (or B7);
in-context scores at B1 or B3 would see fewer examples than at B15, so this is
an upper bound for the early budgets.
Direct B15 check on the same appearances: the score as a logit offset on hier's
B15 prediction for the evaluated items, q = sigmoid(logit p + cap(beta (x -
mean x))), x centred over the pair's evaluated items; beta either per pair (the
harness's pair_slope MAP from the 15 labeled items, u = hier's unlabeled-twin
prediction, c = the cross-fitted icl_cf or the zs scores centred over the
labeled items, prior N(0, (S_PP / sd)^2), sd the pooled within-pair sd of the
evaluated scores) or transferred (one beta per held-out parent, fitted on the
other parents' appearances by harness.fit_scalar on their mean B15 Brier).
Reported: the B15 Brier difference against hier, and the same for the honest
easiness as x (the oracle on these appearances). One budget, 16 appearances:
a direction check for the value map, not an ALC.

Run:  python experiments/icl_probe.py --stage select            # seconds, CPU
      nohup python experiments/icl_probe.py --stage run --passes core > LOG 2>&1 &
                          # 41 min for 568,168 prompt tokens (231/s), the only LLM job, int8 4 GB
      python experiments/icl_probe.py --stage run --passes perm,half   # 51 min, 624,729 tokens
      python experiments/icl_probe.py --stage analyse             # seconds, CPU
      python experiments/icl_probe.py --stage show
Needs data/<benchmark>/, the harness rows (python experiments/harness.py --stage
collect) and, for --stage run only, Qwen/Qwen3-4B-Instruct-2507 in the local
Hugging Face cache.
"""
import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
# cap the MPS pool (attempt_probe's setting): past it an allocation fails instead of swapping
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import harness as H  # noqa: E402

OUT = os.path.join(ROOT, "results", "icl_probe.json")
DATA = os.path.join(ROOT, "data", "icl_probe")
PLAN_PATH = os.path.join(DATA, "plan.json")
SCORES = os.path.join(DATA, "scores")
PARENTS = H.PARENTS
B = 15
BI = H.BUDGETS.index(B)
PER_PARENT = 4
MAX_EVAL = 45
MIN_EVAL = 20
MIN_CLASS = 4
TEXT_SHARE = 0.8
SEED = 20260927
EX_TOKENS, TGT_TOKENS = 192, 384
KILL_GAIN, KILL_R = 0.1, 0.3
S_PP = 0.5                      # per-pair prior sd per within-pair sd of the score (the harness's s)
PASSES = ("core", "perm", "half")
BOOTS = 2000
OTHER_LLM = ("probe_attempts|llm_|lora|icl_probe|probe_compare|hidden_state|pairwise_probe|"
             "attempt_probe.py --stage gen")
VALUE_LINES = ("transferred nested", "per-pair nested", "transferred from B1 (forced)",
               "transferred from B7 (forced)", "per-pair s=0.5 from B7 (forced)")

SYSTEM = "You are an expert at predicting which tasks a particular AI system can and cannot solve."
INTRO = ("The tasks below all come from one benchmark. One AI system attempted each of them once. "
         "After each task you see whether that system SOLVED it or FAILED it. Long tasks are "
         "shortened, with the middle omitted.\n\n")
EX_OPEN = "<example>\n<task>\n"
EX_CLOSE = "\n</task>\nOutcome: {outcome}\n</example>\n\n"
BRIDGE = ("Study which kinds of tasks this system solves and which it fails. Then judge a new task "
          "from the same benchmark.\n\n<task>\n")
BRIDGE_ZS = ("Below is a task from an AI evaluation benchmark that one AI system attempted once. "
             "Long tasks are shortened, with the middle omitted.\n\n<task>\n")
SUFFIX = "\n</task>\n\nDid this system solve this task? Answer Yes or No."
_BODY, _TGT = "\u0000BODY\u0000", "\u0000TGT\u0000"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hh(*a):
    return int(hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()[:12], 16)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def other_llm_processes(pattern=OTHER_LLM):
    """Other processes whose command line matches `pattern`; this one and its
    ancestors (a shell chain that names this script) excluded."""
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
    import re
    pat = re.compile(pattern)
    return [f"{p} {c[:160]}" for p, (_, c) in procs.items()
            if p not in mine and pat.search(c) and "pgrep" not in c and "grep " not in c
            and not re.match(r"(\S*/)?(ba|z|da)?sh\b", c)]


def load_items(parents=PARENTS):
    """{parent: {item_id: {item_content, item_features}}} from data/<parent>/items.parquet."""
    import pandas as pd
    out = {}
    for b in parents:
        it = pd.read_parquet(os.path.join(ROOT, "data", b, "items.parquet"),
                             columns=["item_id", "content", "item_features"])
        out[b] = {str(i): {"item_content": "" if c is None else str(c),
                           "item_features": "" if f is None else str(f)}
                  for i, c, f in zip(it.item_id, it.content, it.item_features)}
    return out


# --- selection --------------------------------------------------------------------------

def stage_select(args):
    from experiments import llm4b_close as L4
    from paiec import itemcov as IC
    items = load_items()
    # matharena: llm4b_close.text_bearing (no image placeholder, >= 70 characters: shorter
    # content is a system prompt or a date and an id); elsewhere short tasks are real
    # ("Order me an apron and a pot for cooking."), so only image placeholders are dropped
    textual = {b: (L4.text_bearing(v) if b == "matharena" else
                   {k: bool(IC.image_ref(t) == 0 and t["item_content"].strip()) for k, t in v.items()})
               for b, v in items.items()}
    apps, counts = [], {p: {"appearances": 0, "eligible": 0} for p in PARENTS}
    for i in range(H.PLAN["tl"]):
        path = H.row_path(H.ROWS, "tl", i)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            r = pickle.load(f)
        for j, s in enumerate(r["slots"]):
            par = s["parent"]
            counts[par]["appearances"] += 1
            y = np.asarray(s["acq_y"][:B], int)
            ev = np.asarray(s["ev_keys"], object)
            if len(y) < B or not 2 <= y.sum() <= B - 2 or len(ev) < MIN_EVAL:
                continue
            order = sorted(range(len(ev)), key=lambda k: hh(SEED, "ev", s["sid"], ev[k]))[:MAX_EVAL]
            K, N = np.asarray(s["ev_K"], float)[order], np.asarray(s["ev_N"], float)[order]
            yk = K / np.maximum(N, 1)
            if (yk > 0.5).sum() < MIN_CLASS or (yk < 0.5).sum() < MIN_CLASS:
                continue
            keys = list(ev[order]) + [str(k) for k in s["acq_keys"][:B]]
            if np.mean([textual[par].get(str(k), False) for k in keys]) < TEXT_SHARE:
                continue
            counts[par]["eligible"] += 1
            apps.append(dict(id=f"tl{i}_{j}", run=i, slot=j, parent=par, kind=s["kind"], name=s["name"],
                             sid=s["sid"], subject=s["subject"].get("normalized_name"),
                             ev_keys=[str(k) for k in ev[order]], ev_K=K.tolist(), ev_N=N.tolist(),
                             ev_p=np.asarray(s["ev_p"][BI], float)[order].tolist(),
                             lab_keys=[str(k) for k in s["acq_keys"][:B]], lab_y=y.tolist(),
                             lab_pu=np.asarray(s["acq_pu"][BI][:B], float).tolist(),
                             n_eval_all=int(len(ev))))
    sel = []
    for par in PARENTS:
        cand = sorted([a for a in apps if a["parent"] == par], key=lambda a: hh(SEED, "pick", a["id"]))
        seen = set()
        for a in cand:
            if a["sid"] in seen:
                continue
            seen.add(a["sid"])
            sel.append(a)
            if len(seen) == PER_PARENT:
                break
    log("honest difficulty (Rasch per parent and subject fold) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, honest, _ = H.oracle_maps()
    for a in sel:
        m = honest[H.fold_of(a["sid"])]
        a["fold"] = int(H.fold_of(a["sid"]))
        a["ev_b"] = [m.get(k, None) for k in a["ev_keys"]]
        a["lab_b"] = [m.get(k, None) for k in a["lab_keys"]]
        halves = sorted(range(B), key=lambda k: hh(SEED, "half", a["id"], a["lab_keys"][k]))
        a["lab_half"] = [0] * B
        for r_, k in enumerate(halves):
            a["lab_half"][k] = r_ % 2
        perm = sorted(range(B), key=lambda k: hh(SEED, "perm", a["id"], k))
        a["perm_y"] = [a["lab_y"][k] for k in perm]
    plan = dict(budget=B, per_parent=PER_PARENT, max_eval=MAX_EVAL, seed=SEED, counts=counts,
                eligible_total=len(apps), appearances=sel,
                harness_rows_lib=sorted({str(pickle.load(open(H.row_path(H.ROWS, "tl", a["run"]), "rb"))
                                             .get("lib_digest")) for a in sel}))
    write_json(PLAN_PATH, plan)
    for a in sel:
        log(a["id"], a["parent"], a["kind"], a["subject"], f"labels {sum(a['lab_y'])}/{B}",
            f"eval {len(a['ev_keys'])} rate {np.sum(a['ev_K']) / np.sum(a['ev_N']):.2f}")
    log("eligible per parent", {p: c["eligible"] for p, c in counts.items()}, "of",
        {p: c["appearances"] for p, c in counts.items()})


# --- the language model ------------------------------------------------------------------

class Scorer:
    """Yes/No logits after a shared prompt prefix (its KV cache reused per target)."""

    def __init__(self):
        import torch
        from paiec import llmfeat as F
        from experiments.attempt_probe import load_model
        self.torch, self.F = torch, F
        t0 = time.time()
        self.model, self.tok, self.rev = load_model("int8")
        log(f"model loaded in {time.time() - t0:.0f}s (revision {self.rev})")
        self.yes = F.encode(self.tok, "Yes")
        self.no = F.encode(self.tok, "No")
        assert len(self.yes) == len(self.no) == 1, "Yes/No are not single tokens"
        self.yes, self.no = self.yes[0], self.no[0]
        self.marker = F.encode(self.tok, F.MARKER)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _BODY + _TGT + SUFFIX}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        head, rest = text.split(_BODY)
        mid, tail = rest.split(_TGT)
        assert mid == "", "template put text between body and target"
        self.rendered = text
        self.head, self.tail = F.encode(self.tok, head), F.encode(self.tok, tail)
        self.cache_ids = {}
        self.tokens = 0

    def ids(self, text, n):
        key = (text, n)
        if key not in self.cache_ids:
            self.cache_ids[key], _ = self.F.head_tail(self.F.encode(self.tok, text), n, self.marker)
        return self.cache_ids[key]

    def prefix(self, examples):
        """examples: [(text, outcome 0/1)]; [] is the zero-shot control."""
        F = self.F
        if not examples:
            return self.head + F.encode(self.tok, BRIDGE_ZS)
        body = F.encode(self.tok, INTRO)
        for t, y in examples:
            body += F.encode(self.tok, EX_OPEN) + self.ids(t, EX_TOKENS) + \
                F.encode(self.tok, EX_CLOSE.format(outcome="SOLVED" if y else "FAILED"))
        return self.head + body + F.encode(self.tok, BRIDGE)

    def score(self, prefix, targets):
        """[(log p(Yes) - log p(No), p(Yes) + p(No))] per target text."""
        torch = self.torch
        res = []
        with torch.inference_mode():
            pids = torch.tensor([prefix], device="mps")
            out = self.model(input_ids=pids, use_cache=True, logits_to_keep=1)
            cache, n0 = out.past_key_values, len(prefix)
            self.tokens += n0
            for t in targets:
                ids = self.ids(t, TGT_TOKENS) + self.tail
                o = self.model(input_ids=torch.tensor([ids], device="mps"), past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
                lp = torch.log_softmax(o.logits[0, -1].float(), -1)
                a, b = float(lp[self.yes]), float(lp[self.no])
                res.append((a - b, math.exp(a) + math.exp(b)))
                cache.crop(n0)
                self.tokens += len(ids)
            del cache, out
        torch.mps.empty_cache()
        return res


def score_path(app_id, pas):
    return os.path.join(SCORES, f"{app_id}_{pas}.json")


def run_pass(S, a, pas, text):
    """{variant: {item_key: [score, yes+no mass]}} for one appearance and pass."""
    ev = a["ev_keys"]
    lab = list(zip(a["lab_keys"], a["lab_y"]))
    halves = [[(k, y) for (k, y), h in zip(lab, a["lab_half"]) if h == f] for f in (0, 1)]
    out, meta = {}, {}

    def put(variant, keys, prefix):
        sc = S.score(prefix, [text[k] for k in keys])
        out.setdefault(variant, {}).update({k: [round(v, 5), round(m, 5)] for k, (v, m) in zip(keys, sc)})
        meta.setdefault(variant + "_prefix_tokens", []).append(len(prefix))

    if pas == "core":
        put("icl", ev, S.prefix([(text[k], y) for k, y in lab]))
        put("zs", ev + a["lab_keys"], S.prefix([]))
        for f in (0, 1):             # labeled items of half f scored with the other half's examples
            put("icl_cf", [k for k, _ in halves[f]], S.prefix([(text[k], y) for k, y in halves[1 - f]]))
    elif pas == "perm":
        put("perm", ev, S.prefix([(text[k], y) for k, y in zip(a["lab_keys"], a["perm_y"])]))
    elif pas == "half":
        for f in (0, 1):
            put(f"half{f}", ev, S.prefix([(text[k], y) for k, y in halves[f]]))
    return out, meta


def stage_run(args):
    plan = read_json(PLAN_PATH)
    if plan is None:
        raise SystemExit("run --stage select first")
    others = other_llm_processes()
    if others and not args.force:
        raise SystemExit("another language-model process is alive:\n  " + "\n  ".join(others))
    passes = [p for p in args.passes.split(",") if p]
    todo = [(p, a) for p in passes for a in plan["appearances"] if not os.path.exists(score_path(a["id"], p))]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        log("nothing to do")
        return
    from paiec import llmfeat as F
    items = load_items()
    text = {}
    for a in plan["appearances"]:
        for k in a["ev_keys"] + a["lab_keys"]:
            text[k] = F.item_text(items[a["parent"]][k])
    S = Scorer()
    t_all = time.time()
    for n, (pas, a) in enumerate(todo):
        t0, tok0 = time.time(), S.tokens
        out, meta = run_pass(S, a, pas, text)
        secs = time.time() - t0
        write_json(score_path(a["id"], pas), dict(id=a["id"], pass_=pas, scores=out, meta=meta,
                                                  secs=round(secs, 1), tokens=S.tokens - tok0,
                                                  model_revision=S.rev, precision="int8"))
        log(f"{n + 1}/{len(todo)} {pas} {a['id']} {a['parent']}: {S.tokens - tok0} tokens in {secs:.0f}s "
            f"({(S.tokens - tok0) / max(secs, 1e-9):.0f}/s); elapsed {time.time() - t_all:.0f}s")
    info = read_json(os.path.join(DATA, "prompt.json")) or {}
    info.update(rendered=S.rendered, system=SYSTEM, intro=INTRO, example=EX_OPEN + "{task}" + EX_CLOSE,
                bridge=BRIDGE, bridge_zs=BRIDGE_ZS, suffix=SUFFIX, ex_tokens=EX_TOKENS, tgt_tokens=TGT_TOKENS,
                model_revision=S.rev, precision="int8")
    write_json(os.path.join(DATA, "prompt.json"), info)


# --- analysis ----------------------------------------------------------------------------

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def ranks(x):
    """Average ranks (ties share their mean rank); NaN stays NaN."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(np.isfinite(x))
    v = x[ok]
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    out[ok] = (np.bincount(inv, r) / cnt)[inv]
    return out


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pos, neg = x[y > 0.5], x[y < 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def partial(x, y, z):
    """Pearson of x and y, both net of a linear fit on z."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    if np.std(z) < 1e-9:
        return pearson(x, y)
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearson(rx, ry)


def load_scores(plan):
    got = {}
    for a in plan["appearances"]:
        s = {}
        for p in PASSES:
            d = read_json(score_path(a["id"], p))
            if d is not None:
                for v, m in d["scores"].items():
                    s[v] = {k: t[0] for k, t in m.items()}
                    s[v + "_mass"] = {k: t[1] for k, t in m.items()}
                s["_secs_" + p], s["_tokens_" + p] = d["secs"], d["tokens"]
        got[a["id"]] = s
    return got


def per_app(a, s):
    """Scores and references over the evaluated items of one appearance."""
    ev = a["ev_keys"]
    y = np.array(a["ev_K"]) / np.maximum(np.array(a["ev_N"]), 1)
    d = np.array([-v if v is not None else np.nan for v in a["ev_b"]], float)
    lp = H.logit(np.clip(np.array(a["ev_p"]), 1e-6, 1 - 1e-6))
    X = {}
    for v in ("icl", "zs", "perm"):
        if v in s:
            X[v] = np.array([s[v][k] for k in ev])
    if "half0" in s and "half1" in s:
        X["half"] = np.array([(s["half0"][k] + s["half1"][k]) / 2 for k in ev])
    X["oracle"] = d
    X["hier"] = lp
    row = {}
    for v, x in X.items():
        okd = np.isfinite(d)
        row[v] = dict(r_diff=pearson(x, d) if v != "oracle" else 1.0, r_out=pearson(x, y), auc=auc(x, y),
                      r_diff_partial=partial(x[okd], d[okd], lp[okd]) if v not in ("oracle", "hier")
                      else float("nan"),
                      r_out_partial=partial(x, y, lp) if v != "hier" else float("nan"),
                      spearman_diff=pearson(ranks(x), ranks(d)) if v != "oracle" else 1.0,
                      spearman_out=pearson(ranks(x), ranks(y)))
    return row, X, y, d, lp


def summarise(vals):
    """mean, SE over appearances, SE across parent means, per-parent means."""
    v = np.array([x for _, x in vals], float)
    par = np.array([p for p, _ in vals], object)
    ok = np.isfinite(v)
    pm = {p: float(np.mean(v[ok & (par == p)])) for p in PARENTS if (ok & (par == p)).any()}
    pmv = np.array(list(pm.values()))
    return dict(mean=round(float(np.mean(v[ok])), 4) if ok.any() else None,
                se=round(float(np.std(v[ok], ddof=1) / math.sqrt(ok.sum())), 4) if ok.sum() > 1 else None,
                parent_se=round(float(np.std(pmv, ddof=1) / math.sqrt(len(pmv))), 4) if len(pmv) > 1 else None,
                n=int(ok.sum()), per_parent={p: round(x, 4) for p, x in pm.items()})


def value_map(r):
    """The honest gate table read at a within-pair r (linear interpolation over
    r_within_pair_tl; below the r = 0 row it reads that row)."""
    thr = read_json(os.path.join(ROOT, "results", "harness_thresholds.json"))
    rows = sorted(thr["tables"]["honest"].values(), key=lambda t: t["meta"]["r_within_pair_tl"])
    xs = np.array([max(t["meta"]["r_within_pair_tl"], 0.0) for t in rows])
    out = {}
    for ln in VALUE_LINES:
        ys = np.array([t["lines"][ln]["regimes"]["tl"]["est"] for t in rows])
        out[ln] = round(float(np.interp(max(r, 0.0), xs, ys)), 5)
    return out


def b15_check(plan, scores, variant):
    """B15 Brier difference against hier on the evaluated items, x = the score
    (or the honest easiness for 'oracle'), per-pair and transferred slopes."""
    need = ("half0", "half1") if variant == "half" else (variant,)
    apps = [a for a in plan["appearances"] if variant == "oracle" or all(v in scores[a["id"]] for v in need)]
    if len(apps) < 8:
        return None
    E = []
    for a in apps:
        s = scores[a["id"]]
        _, X, y, d, lp = per_app(a, s)
        x = X["oracle" if variant == "oracle" else variant]
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        if variant == "oracle":
            lb = np.array([-v if v is not None else np.nan for v in a["lab_b"]], float)
            cl = np.where(np.isfinite(lb), lb, np.nanmean(lb))
        elif variant == "icl":
            cl = np.array([s["icl_cf"][k] for k in a["lab_keys"]]) if "icl_cf" in s else None
        elif variant == "zs":
            cl = np.array([s["zs"][k] for k in a["lab_keys"]])
        else:
            cl = None
        E.append(dict(parent=a["parent"], x=x - x.mean(), lp=lp, K=np.array(a["ev_K"], float),
                      N=np.array(a["ev_N"], float), cl=None if cl is None else cl - cl.mean(),
                      u=H.logit(np.clip(np.array(a["lab_pu"]), 1e-6, 1 - 1e-6)), yl=np.array(a["lab_y"], float)))
    sd = math.sqrt(np.mean(np.concatenate([e["x"] ** 2 for e in E])))
    sd = sd if sd > 0 else 1.0

    def dbrier(e, beta):
        p = H.sigmoid(e["lp"])
        q = H.sigmoid(e["lp"] + H.capped(beta * e["x"]))
        return float(np.sum(e["N"] * (q * q - p * p) - 2 * e["K"] * (q - p)) / np.sum(e["N"]))

    out = {}
    tr = []
    for e in E:
        train = [f for f in E if f["parent"] != e["parent"]]
        beta = H.fit_scalar(lambda b: float(np.mean([dbrier(f, b) for f in train])), 1 / sd)
        tr.append((e["parent"], dbrier(e, beta), beta))
    out["transferred"] = dict(summarise([(p, v) for p, v, _ in tr]),
                              beta_per_parent={p: round(b, 4) for p, _, b in tr})
    if all(e["cl"] is not None for e in E):
        pp = []
        for e in E:
            m = np.ones((1, len(e["yl"])), bool)
            beta = float(H.pair_slope(e["u"][None], e["yl"][None], e["cl"][None], m, 0.0, S_PP / sd)[0])
            pp.append((e["parent"], dbrier(e, beta), beta))
        out["per-pair"] = dict(summarise([(p, v) for p, v, _ in pp]),
                               beta_mean=round(float(np.mean([b for _, _, b in pp])), 4))
    out["sd"] = round(sd, 4)
    return out


def stage_analyse(args):
    plan = read_json(PLAN_PATH)
    scores = load_scores(plan)
    rows, meta = [], {}
    for a in plan["appearances"]:
        s = scores[a["id"]]
        if "icl" not in s:
            continue
        row, X, y, d, lp = per_app(a, s)
        mass = {v: round(float(np.mean([s[v + "_mass"][k] for k in a["ev_keys"]])), 4)
                for v in ("icl", "zs", "perm", "half0", "half1") if v + "_mass" in s}
        level = {v: round(float(np.mean(H.sigmoid(X[v]))), 4) for v in ("icl", "zs", "perm", "half") if v in X}
        rows.append(dict(id=a["id"], parent=a["parent"], kind=a["kind"], subject=a["subject"],
                         lab_rate=round(float(np.mean(a["lab_y"])), 3),
                         ev_rate=round(float(np.sum(a["ev_K"]) / np.sum(a["ev_N"])), 3),
                         hier_b15_mean=round(float(np.mean(H.sigmoid(lp))), 4), n_eval=len(a["ev_keys"]),
                         stats={v: {k: (round(x, 4) if x == x else None) for k, x in st.items()}
                                for v, st in row.items()},
                         yes_no_mass=mass, mean_p_yes=level,
                         secs={p: s.get("_secs_" + p) for p in PASSES}, tokens={p: s.get("_tokens_" + p)
                                                                              for p in PASSES}))
    if not rows:
        raise SystemExit("no scores yet")
    variants = [v for v in ("icl", "zs", "perm", "half", "hier", "oracle") if all(v in r["stats"] for r in rows)]
    table = {}
    for v in variants:
        table[v] = {m: summarise([(r["parent"], r["stats"][v][m] if r["stats"][v][m] is not None else np.nan)
                                  for r in rows]) for m in ("r_diff", "r_out", "auc", "r_out_partial", "r_diff_partial", "spearman_diff",
                                           "spearman_out")}
    diffs = {}
    for a_, b_ in (("icl", "zs"), ("icl", "perm"), ("icl", "half"), ("half", "zs"), ("perm", "zs")):
        if a_ in variants and b_ in variants:
            diffs[f"{a_} - {b_}"] = {m: summarise([(r["parent"], (r["stats"][a_][m] or np.nan) -
                                                    (r["stats"][b_][m] or np.nan)) for r in rows])
                                     for m in ("r_diff", "r_out", "auc", "r_out_partial", "r_diff_partial",
                                               "spearman_diff", "spearman_out")}
    # level: does the prompt carry the subject's rate across appearances?
    lev = {}
    for v in ("icl", "zs", "perm", "half"):
        if all(v in r["mean_p_yes"] for r in rows):
            lev[v] = round(pearson([r["mean_p_yes"][v] for r in rows], [r["ev_rate"] for r in rows]), 4)
    lev["hier_b15"] = round(pearson([r["hier_b15_mean"] for r in rows], [r["ev_rate"] for r in rows]), 4)
    lev["label_rate"] = round(pearson([r["lab_rate"] for r in rows], [r["ev_rate"] for r in rows]), 4)
    r_icl = table["icl"]["r_diff"]["mean"]
    r_zs = table["zs"]["r_diff"]["mean"]
    implied = {v: value_map(table[v]["r_diff"]["mean"]) for v in ("icl", "zs", "perm", "half") if v in table}
    b15 = {v: b15_check(plan, scores, v) for v in ("icl", "zs", "perm", "half", "oracle")}
    kill = (r_icl - r_zs < KILL_GAIN) or (r_icl < KILL_R)
    verdict = dict(rule=f"KILL if r_diff(icl) - r_diff(zs) < {KILL_GAIN} or r_diff(icl) < {KILL_R} "
                        "(mean within-pair Pearson with honest easiness over the evaluated items)",
                   r_icl=r_icl, r_zs=r_zs, gain=round(r_icl - r_zs, 4),
                   r_out_icl=table["icl"]["r_out"]["mean"], r_out_zs=table["zs"]["r_out"]["mean"],
                   call="KILL" if kill else "CONTINUE")
    tot = {p: dict(secs=round(sum(r["secs"][p] or 0 for r in rows), 1),
                   tokens=int(sum(r["tokens"][p] or 0 for r in rows))) for p in PASSES}
    with open(os.path.abspath(__file__), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    res = dict(meta=dict(script="experiments/icl_probe.py", script_digest=digest, budget=B, per_parent=PER_PARENT,
                         max_eval=MAX_EVAL, seed=SEED, ex_tokens=EX_TOKENS, tgt_tokens=TGT_TOKENS,
                         selection=dict(counts=plan["counts"], eligible_total=plan["eligible_total"],
                                        rule=f"15 own labels with 2..13 successes; >= {MIN_EVAL} evaluated items "
                                             f"(subsampled to {MAX_EVAL}), >= {MIN_CLASS} solved and failed; "
                                             f">= {TEXT_SHARE:.0%} text-bearing"),
                         harness_rows_lib=plan["harness_rows_lib"], prompt=read_json(os.path.join(DATA, "prompt.json")),
                         compute=tot, s_pp=S_PP),
               verdict=verdict, table=table, paired=diffs, level_r_across_appearances=lev,
               implied_alc=implied, b15_check=b15, appearances=rows)
    write_json(OUT, res)
    show(res)


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def show(res):
    t = res["table"]
    print("\n| score | r_diff (± SE / parent SE) | r_out | AUC | r_out net of hier B15 | per parent r_diff |")
    print("|---|---|---|---|---|---|")
    for v, st in t.items():
        rd = st["r_diff"]
        print(f"| {v} | {_f(rd['mean'])} ± {rd['se']} / {rd['parent_se']} | {_f(st['r_out']['mean'])} | "
              f"{st['auc']['mean']:.3f} | {_f(st['r_out_partial']['mean'])} | "
              + ", ".join(f"{p[:5]} {_f(x, 2)}" for p, x in rd["per_parent"].items()) + " |")
    print("\npaired:", json.dumps({k: {m: (v[m]["mean"], v[m]["se"]) for m in v} for k, v in res["paired"].items()}))
    print("level r across appearances:", res["level_r_across_appearances"])
    print("implied ALC:", json.dumps(res["implied_alc"]))
    print("B15 check:", json.dumps({k: (None if v is None else {m: (v[m]["mean"], v[m]["se"]) if isinstance(v[m], dict)
                                                               and "mean" in v[m] else v[m] for m in v})
                                    for k, v in res["b15_check"].items()}))
    print("verdict:", json.dumps(res["verdict"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["select", "run", "analyse", "show"])
    ap.add_argument("--passes", default="core")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="run beside another language-model process")
    args = ap.parse_args()
    if args.stage == "select":
        stage_select(args)
    elif args.stage == "run":
        stage_run(args)
    elif args.stage == "analyse":
        stage_analyse(args)
    else:
        res = read_json(OUT)
        if res is None:
            raise SystemExit("run --stage analyse first")
        show(res)


if __name__ == "__main__":
    main()
