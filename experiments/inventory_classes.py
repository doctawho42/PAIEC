"""Task-type shares of the organisers' benchmark inventory, from titles alone.

The hidden benchmarks are drawn from the organisers' inventory of candidate
benchmarks (the Google Sheet linked from the competition page; candidates are
assigned at random to the training and test pools before curation). This
classifies each title by keyword rules into 'not an AI-system evaluation'
(vision, RL, science datasets and the like) or one of text_qa, math, code,
agent, mm_image, mm_video, audio, and checks the rules against 80 labels made
by reading titles. The classes are read from the TITLE only: no description,
paper, repository or item is read. Nothing here enters a model, a prior or a
selection; the shares are used only to say how the public benchmarks differ
from the hidden pool (docs/report/draft.md section 2.1; docs/findings.md "What
this could do on the hidden test").

Moved from session scratch (rethink2/methodology-critic/c_inventory_meta.py,
part C1, and rethink2/attempt-signals/a6_inventory_formats.py,
a6b_inventory_llm_only.py, a6c_inventory_safety.py, 2026-09-26); the rules are
unchanged. Not moved: c_inventory_meta.py's C2 (a random-effects meta-analysis
of per-benchmark correlations) and C3 (units needed for a CI), which no
documented number uses.

Inputs:
  results/inventory.csv              the 161-row inventory (the sheet's first
                                     161 rows, all NeurIPS 2025 Datasets and
                                     Benchmarks papers; committed 2026-09-23)
  results/inventory_hand_labels.csv  80 titles sampled from the 1,261-row sheet
                                     (DataFrame.sample(80, random_state=11)),
                                     labelled llm_eval (0/1) and a category by
                                     reading each title. The labels were written
                                     by an AI agent lane of the research session
                                     (Claude), not by a person; the file records
                                     no other author.
  --sheet (optional)                 the 1,261-row sheet as downloaded on
                                     2026-09-24 (default data/inventory_sheet.csv,
                                     gitignored; sha256 recorded in the results).
                                     The competition page says the collections
                                     may grow, so a new download can differ.

Validation (C1): agreement on llm_eval over all 80 labels, and on the category
over the labels both call an AI-system evaluation. The category shares are
shares of the titles classified as AI-system evaluations, not of all titles.

The a6 classes (agentic, multimodal, code, math, qa_reason, other; first match
wins) ask a different question: which benchmarks a local model could attempt
from the item text with a short, checkable answer. They are reported for the
161 rows and, with --sheet, for the 1,261.

    python experiments/inventory_classes.py        # seconds; no network

Output: results/inventory_classes.json and results/inventory_classes.csv (the
161 rows with their classes). --out and --out-csv write elsewhere, to check a
re-run against the stored files without overwriting them.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV161 = os.path.join(ROOT, "results", "inventory.csv")
HAND = os.path.join(ROOT, "results", "inventory_hand_labels.csv")
SHEET = os.path.join(ROOT, "data", "inventory_sheet.csv")
OUT = os.path.join(ROOT, "results", "inventory_classes.json")
OUT_CSV = os.path.join(ROOT, "results", "inventory_classes.csv")

# --- C1: AI-system evaluation or not, and its category (c_inventory_meta.py) ------------

NOT_LLM = r"segmentation|tracking|reconstruct|pose estimation|optical flow|\bCFD\b|fluid dynamics|point cloud|depth estimation|forecast|avatar|diffusion model|image editing|editing model|video generation|subject-to-video|text-to-image|image generation|neural field|\bRNA\b|protein|EMG|electromyograph|physiolog|reinforcement learning|off-policy|scheduling|few-shot classification|out-of-distribution detection|OOD detection|watermark|dataset for|aerodynamic|MRI|\bPET\b|remote sensing|lidar|\bUAV\b|autonomous driving|trajectory prediction|time series|time-series|graph neural|tabular|single-cell|\bcell\b|molecul|materials|deepfake|fake image|anomaly detection|re-identification|human motion|TTS|speech synthesis|world model|robot manipulation|robotic manipulation|vision-language-action|VLA"
LLM = r"\bLLMs?\b|language models?|\bMLLMs?\b|\bVLMs?\b|\bLMMs?\b|vision[- ]language|multimodal large|foundation models?|\bagents?\b|agentic|question answering|\bQA\b|VQA|reasoning|chatbot|GPT|hallucinat|benchmarking .*models|can .* models|jailbreak|red[- ]team|instruction|prompt|exam|olympiad|coding|code generation|software engineering|issue resolving"
CATS = [
    ("agent", r"\bagents?\b|agentic|\bGUI\b|web ?agent|websites?|tool[- ]use|tool use|computer[- ]use|embodied|interactive|\bOS\b|mobile|environment|\bgame|planning|simulat"),
    ("code", r"\bcode\b|coding|program|software|\bSWE\b|SWE-|repositor|compiler|verilog|\bSQL\b|informatics|decompil|vulnerab|kernel|\bbug"),
    ("mm_video", r"video"),
    ("audio", r"audio|speech|spoken|music|sound|acoustic"),
    ("mm_image", r"vision|visual|image|\bVQA\b|multimodal|multi-modal|\bMLLM|\bVLM|\bLMM|chart|diagram|document|\b3D\b|spatial|OCR|caption|photo|pixel|scene"),
    ("math", r"\bmath|theorem|proof|olympiad|geometr|physics|arithmetic|inequalit|calcul|algebra"),
]
STATIC = ["text_qa", "math", "mm_image", "mm_video", "audio"]


def classify(title):
    t = str(title)
    llm = re.search(LLM, t, re.I) is not None
    notllm = re.search(NOT_LLM, t, re.I) is not None
    if notllm and not llm:
        return 0, ""
    if not llm and not re.search(r"bench|benchmark|evaluat", t, re.I):
        return 0, ""
    for c, pat in CATS:
        if re.search(pat, t, re.I):
            return 1, c
    return 1, "text_qa"


def shares(df):
    lab = df.title.apply(classify)
    df = df.assign(llm_eval=[a for a, _ in lab], cat=[b for _, b in lab])
    ev = df[df.llm_eval == 1]
    return df, dict(n=int(len(df)), n_llm_eval=int(len(ev)), share_llm_eval=round(len(ev) / len(df), 4),
                    cat_counts={k: int(v) for k, v in ev.cat.value_counts().items()},
                    cat_share=ev.cat.value_counts(normalize=True).round(4).to_dict(),
                    static_qa_like=round(float(ev.cat.isin(STATIC).mean()), 4),
                    text_or_math=round(float(ev.cat.isin(["text_qa", "math"]).mean()), 4),
                    code_or_agent=round(float(ev.cat.isin(["code", "agent"]).mean()), 4))


def validate(hand):
    lab = hand.title.apply(classify)
    hand = hand.assign(kw_llm=[a for a, _ in lab], kw_cat=[b for _, b in lab])
    both = hand[(hand.llm_eval == 1) & (hand.kw_llm == 1)]
    coarse = lambda c: "codeagent" if c in ("code", "agent") else "static"
    hand_ev = hand[hand.llm_eval == 1]
    k, n = int(hand_ev.cat.isin(STATIC).sum()), len(hand_ev)
    lo, hi = stats.beta.ppf([0.025, 0.975], [k, k + 1], [n - k + 1, n - k])
    return dict(n=int(len(hand)),
                agree_llm_eval=float((hand.kw_llm == hand.llm_eval).mean()),
                n_llm_eval_by_labels=int(len(hand_ev)), n_llm_eval_by_both=int(len(both)),
                agree_cat=float((both.kw_cat == both.cat).mean()),
                agree_cat_count=[int((both.kw_cat == both.cat).sum()), int(len(both))],
                agree_static_vs_codeagent=float((both.kw_cat.map(coarse) == both.cat.map(coarse)).mean()),
                labels_share_llm_eval=float(hand.llm_eval.mean()),
                labels_cat_share=hand_ev.cat.value_counts(normalize=True).round(4).to_dict(),
                labels_static_share_ci95=[float(lo), float(hi), k, n],
                in_first_161_rows=int((hand.sheet_row < 161).sum()),
                confusion={str(a): {str(b): int(v) for b, v in row.items()}
                           for a, row in pd.crosstab(both.cat, both.kw_cat).to_dict().items()})


# --- a6: which benchmarks a local model could attempt (a6_inventory_formats.py) --------

A6 = [
    ("agentic", r"\bagent|web\b|website|browser|\bgui\b|computer[- ]use|\btool|embodied|robot|\bgame|environment|interactive|navigation|planning|\bos\b|desktop|mobile app|smartphone"),
    ("multimodal", r"image|vision|visual|video|\bvlm|mllm|multimodal|multi-modal|audio|speech|chart|document|\b3d\b|spatial|scene|\bocr\b|diagram|\bmaps?\b|pixel|photo|figure|table|egocentric|sound|music"),
    ("code", r"\bcode|coding|program|software|\bswe|repositor|\bbug|\bsql|compiler|verilog|kernel|\bapi|developer|github|debug"),
    ("math", r"\bmath|theorem|olympiad|proof|arithmetic|geometr|algebra|inequalit|calculus|formal|lean\b|numer"),
    ("qa_reason", r"question|\bqa\b|knowledge|reason|exam|multiple-choice|multiple choice|science|scientific|puzzle|logic|fact|trivia|quiz|understand|medical|clinical|legal|financ|chemi|physic|biolog|commonsense|temporal|causal"),
]
A6_LLM = r"\bllms?\b|language model|\bagents?\b|mllm|\bvlms?\b|\blmms?\b|foundation model|\bai\b|chatbot|gpt|reasoning model"
A6_SAFETY = r"safety|safe\b|jailbreak|harm|toxic|refus|unsafe|red[- ]team|attack|adversarial|privacy|misuse|deception|honest|sycophan|bias|fairness|hallucinat"
A6_GEN = r"generation|writing|summar|dialog|conversation|story|creative|instruction[- ]follow|translation|role[- ]play|persona"


def a6_class(title):
    s = str(title if isinstance(title, str) else "").lower()
    for name, pat in A6:
        if re.search(pat, s):
            return name
    return "other"


def a6(df):
    t = df.title.fillna("").str.lower()
    cls = t.apply(a6_class)
    counts = cls.value_counts()
    llm = t.str.contains(A6_LLM, regex=True)
    safety = t.str.contains(A6_SAFETY, regex=True)
    gen = t.str.contains(A6_GEN, regex=True)
    sub = cls[llm]
    return cls, llm, dict(
        n=int(len(df)), counts={k: int(v) for k, v in counts.items()},
        shares={k: round(v / len(df), 3) for k, v in counts.items()},
        short_checkable_plausible_share=round(float(cls.isin(["math", "qa_reason"]).mean()), 3),
        llm_titles={"n": int(llm.sum()), "share_of_inventory": round(float(llm.mean()), 3),
                    "counts": {k: int(v) for k, v in sub.value_counts().items()},
                    "shares": {k: round(v / len(sub), 3) for k, v in sub.value_counts().items()},
                    "short_checkable_plausible_share": round(float(sub.isin(["math", "qa_reason"]).mean()), 3)},
        safety={"safety_share_llm": round(float((safety & llm).sum() / max(llm.sum(), 1)), 3),
                "gen_share_llm": round(float((gen & llm & ~safety).sum() / max(llm.sum(), 1)), 3),
                "safety_by_class": {k: int(v) for k, v in cls[safety & llm].value_counts().items()}})


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", default=SHEET, help="the 1,261-row inventory sheet (optional)")
    ap.add_argument("--out", default=OUT, help="the results JSON (default results/inventory_classes.json)")
    ap.add_argument("--out-csv", dest="out_csv", default=OUT_CSV,
                    help="the 161 rows with their classes (default results/inventory_classes.csv)")
    a = ap.parse_args()
    inv = pd.read_csv(INV161)
    hand = pd.read_csv(HAND).fillna("")
    out = {"inputs": {"inventory_161": {"path": os.path.relpath(INV161, ROOT), "sha256": sha256(INV161)},
                      "hand_labels": {"path": os.path.relpath(HAND, ROOT), "sha256": sha256(HAND)}}}
    out["validation"] = validate(hand)
    df161, out["inventory_161"] = shares(inv)
    cls161, llm161, out["a6_inventory_161"] = a6(inv)
    if a.sheet and os.path.exists(a.sheet):
        sheet = pd.read_csv(a.sheet)
        out["inputs"]["sheet"] = {"path": os.path.relpath(a.sheet, ROOT), "sha256": sha256(a.sheet), "rows": len(sheet),
                                  "first_161_slugs_match": bool(list(sheet.benchmark_name_slug.head(len(inv)))
                                                                == list(inv.benchmark_name_slug))}
        _, out["inventory_sheet"] = shares(sheet)
        _, _, out["a6_inventory_sheet"] = a6(sheet)
    else:
        out["inputs"]["sheet"] = None
    df161 = df161.assign(a6_class=cls161.values, a6_llm_title=llm161.astype(int).values)
    df161[["benchmark_name_slug", "title", "llm_eval", "cat", "a6_class", "a6_llm_title"]].to_csv(a.out_csv, index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    out["meta"] = {"script_digest": sha256(os.path.abspath(__file__))[:16], "commit": commit,
                   "command": " ".join(["python", "experiments/inventory_classes.py"] + sys.argv[1:]),
                   "provenance": "session scratch rethink2/methodology-critic/c_inventory_meta.py (C1) and "
                                 "rethink2/attempt-signals/a6*_inventory_*.py, 2026-09-26; rules unchanged"}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    v, s = out["validation"], out["inventory_161"]
    print(f"validation: llm_eval agreement {v['agree_llm_eval']:.3f} over {v['n']}; category agreement "
          f"{v['agree_cat']:.3f} ({v['agree_cat_count'][0]} of {v['agree_cat_count'][1]} both call an evaluation)")
    print(f"161 rows: {s['n_llm_eval']} AI-system evaluations ({s['share_llm_eval']:.1%}); shares among them: "
          + ", ".join(f"{k} {x:.1%}" for k, x in s["cat_share"].items()))
    if "inventory_sheet" in out:
        s2 = out["inventory_sheet"]
        print(f"sheet ({s2['n']} rows): {s2['n_llm_eval']} evaluations; " +
              ", ".join(f"{k} {x:.1%}" for k, x in s2["cat_share"].items()))


if __name__ == "__main__":
    main()
