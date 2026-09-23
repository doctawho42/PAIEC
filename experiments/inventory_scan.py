"""List every benchmark repo's files via a blobless shallow clone.

The GitHub API is not enabled for this session (403, add_repo), but git over
HTTPS is. A blobless shallow clone downloads the tree without file contents,
which is all we need to see whether per-item model results were released.
"""
import csv, os, re, shutil, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

RESULT_DIR = re.compile(r"(^|/)(results?|outputs?|predictions?|preds|responses?|"
                        r"eval_results?|model_outputs?|generations?|logs?|submissions?|"
                        r"evaluation|scores?|leaderboard)(/|$)", re.I)
MODEL = re.compile(r"(gpt[-_]?[34o]|gpt4|claude|gemini|llama|qwen|deepseek|mistral|"
                   r"gemma|phi[-_]?[34]|o1[-_]|o3[-_]|grok|command[-_]r|internvl|"
                   r"minicpm|glm[-_]?4|kimi|sonnet|opus|haiku)", re.I)
DATA = re.compile(r"\.(json|jsonl|csv|tsv|parquet|pkl|npy)$", re.I)

lock = threading.Lock()
done = set()
OUT = "clone_scan.csv"


def scan(row):
    slug = row["benchmark_name_slug"]
    url = str(row.get("code_url") or "")
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url)
    rec = dict(slug=slug, repo="", status="no_github", n_files=0, weak=0, strong=0, examples="")
    if not m:
        return rec
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    rec["repo"] = f"{owner}/{repo}"
    d = tempfile.mkdtemp(prefix="bm_")
    try:
        p = subprocess.run(["git", "clone", "--filter=blob:none", "--depth", "1",
                            "--quiet", f"https://github.com/{owner}/{repo}", d],
                           capture_output=True, timeout=180)
        if p.returncode != 0:
            rec["status"] = "clone_fail"
            rec["examples"] = p.stderr.decode()[:100].replace("\n", " ")
            return rec
        ls = subprocess.run(["git", "-C", d, "ls-files"], capture_output=True, timeout=120)
        paths = ls.stdout.decode(errors="replace").splitlines()
        rec["status"] = "ok"
        rec["n_files"] = len(paths)
        weak = [p_ for p_ in paths if DATA.search(p_) and RESULT_DIR.search(p_)]
        strong = [p_ for p_ in weak if MODEL.search(p_)]
        rec["weak"], rec["strong"] = len(weak), len(strong)
        rec["examples"] = " | ".join((strong or weak)[:3])
        return rec
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
        return rec
    except Exception as e:
        rec["status"] = "error"
        rec["examples"] = str(e)[:80]
        return rec
    finally:
        shutil.rmtree(d, ignore_errors=True)


def worker(row):
    rec = scan(row)
    with lock:
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec))
            if new:
                w.writeheader()
            w.writerow(rec)
    return rec


if __name__ == "__main__":
    df = pd.read_csv("inventory.csv")
    seen = set()
    if os.path.exists(OUT):
        seen = set(pd.read_csv(OUT).slug)
    rows = [r for r in df.to_dict("records") if r["benchmark_name_slug"] not in seen]
    print(f"to scan: {len(rows)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(worker, rows), 1):
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    print("done", flush=True)
