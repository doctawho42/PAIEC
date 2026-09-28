"""Write paiec_strong_probe.ipynb from strong_probe.py, so the two never drift.

Plain nbformat-4 JSON, no jupytext. strong_probe.py goes verbatim into a
%%writefile cell (the notebook runs it as a subprocess: a fresh interpreter after
the pip install, vLLM's worker processes outside the Jupyter kernel, and GPU
memory freed when it exits); the other cells start the session clock, install the
pinned vLLM (VLLM_PIN and TRANSFORMERS_PIN, read from the script) and stop the
notebook if the install or `import vllm` fails, define sp() (each command in its
own process group, whose leftovers are killed once it exits: nvidia-smi shows no PIDs
inside Kaggle's container), and call the script's plan, run (with retries that each add the
remedy for the failure just seen) and export commands. tests/test_kaggle_probe.py
checks that the notebook on disk is what this script builds, and executes the
helper and retry cells against fakes and real subprocesses.

    python kaggle/strong_probe/build_notebook.py          # rewrite the notebook
    python kaggle/strong_probe/build_notebook.py --check  # exit 1 if it is stale
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "strong_probe.py")
NOTEBOOK = os.path.join(HERE, "paiec_strong_probe.ipynb")
TARGET = "/tmp/strong_probe.py"          # not /kaggle/working: that directory is the notebook's output

INTRO = """\
# PAIEC K1: более сильная модель на Kaggle — рубрики, попытки решения и энтропия рассуждения

Ноутбук сгенерирован из `kaggle/strong_probe/strong_probe.py` скриптом `build_notebook.py`; правьте скрипт, а не ноутбук.

**Перед запуском** (подробно — `kaggle/strong_probe/README.md`):
1. Settings → Accelerator: **GPU T4 x2** (не P100: compute capability 6.0 ниже минимума vLLM, скрипт сразу остановится).
2. Settings → Internet: **On**.
3. Add-ons → Secrets: секрет **`HF_TOKEN`** (read-токен Hugging Face аккаунта, принявшего условия `aims-foundations/measurement-db`), галочка «attached» у этого ноутбука.
4. Запуск: **Save Version → Save & Run All (Commit)**, не интерактивный прогон: файлы интерактивной сессии не сохраняются как Output версии, а сама сессия умирает при простое или закрытой вкладке. Коммит идёт в фоне, браузер можно закрыть. Сразу после запуска коммита остановите GPU-сессию редактора (кнопка питания / Stop session), иначе она тоже тратит недельную квоту.
5. **Один кусок работы на коммит**, с явными `ARGS` в ячейке ниже: `["--jobs", "rubric"]`, затем `["--jobs", "attempts", "--attempt-scope", "probe"]`, затем `["--jobs", "attempts", "--attempt-scope", "rest"]` (README, «Сессии»); коммит D, энтропия на всех четырёх бенчмарках: `["--jobs", "entropy", "--no-prefix-caching"]` (README, «Коммит D»). Перед Save & Run All проверьте Settings → Quota: осталось ли столько GPU-часов, сколько `plan` пишет в строке `Settings -> Quota shows at least ...` (для D ≈ 10.5); `plan` не видит часов, уже потраченных на этой неделе, а коммит, остановленный квотой, не сохраняет ничего. Output сохраняется, только если коммит завершился: отменённый, упавший или превысивший 12 ч коммит не сохраняет ничего. Падение самого скрипта теряет только шард в работе: ячейка запуска перезапускает его в той же сессии. Скрипт сам останавливается до 11 ч.

**Продолжение после остановки**: Add Input → Your Work → Output предыдущей версии этого ноутбука (в панели Input проверьте, что прикреплена **последняя** версия: прикреплённый Output закреплён на той версии, которую выбрали при добавлении), затем снова Save & Run All. В логе строка `to do:` должна показать меньше задач, чем в прошлый раз. Если Kaggle не даёт прикрепить Output самого ноутбука, сделайте его копию и прикрепите к копии Output оригинала: подойдёт любой `/kaggle/input/**/strong_probe`.

Kaggle сохраняет не больше 500 файлов Output (и монтирует не больше 500 из прикреплённого Output); скрипт держит их около двадцати и сжимает хранилище задолго до предела. Данные measurement-db скачиваются во `/tmp/paiec_data` (не в Output). Токен никуда не печатается и не пишется. В Output попадают признаки, ответы и рассуждения модели и хеши; текстов задач, эталонных ответов и отметок верности там нет. Версию держите приватной."""

CLOCK = """\
# The session clock starts here: the script stops starting shards after --session-hours from this cell.
import os, subprocess, sys, time
os.environ.setdefault("SP_T0", str(time.time()))
print(sys.version)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout)"""

INSTALL = """\
# vLLM {vllm} falls back to its V0 engine on Turing (T4, sm75: plain AWQ GEMM kernel, xformers attention;
# Marlin and FlashAttention-2 need sm80; V0 was removed in 0.11). transformers from the same month.
# It needs Python < 3.13; see README.md for other versions. Either step failing stops the notebook here.
VLLM, TRANSFORMERS = "{vllm}", "{transformers}"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"vllm=={{VLLM}}", f"transformers=={{TRANSFORMERS}}"],
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-4000:])
if r.returncode != 0:
    raise SystemExit(f"pip install failed (exit {{r.returncode}}): see the log above and README.md (other pins)")
r = subprocess.run([sys.executable, "-c", "import vllm, torch, transformers; print('vllm', vllm.__version__, "
                    "'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"],
                   capture_output=True, text=True)
print(r.stdout, r.stderr[-4000:])
if r.returncode != 0:
    raise SystemExit(f"import vllm failed (exit {{r.returncode}}): see the error above and README.md (other pins)")"""

HELPERS = """\
import collections, signal, threading
SCRIPT = "{target}"
# Options for every command: one piece of work per commit (README, "Сессии"), e.g. ["--jobs", "rubric"], then
# ["--jobs", "attempts", "--attempt-scope", "probe"], then ["--jobs", "attempts", "--attempt-scope", "rest"];
# commit D (README, "Коммит D"): ["--jobs", "entropy", "--no-prefix-caching"];
# or ["--model", "Qwen/Qwen3-32B-AWQ", "--jobs", "rubric"], ["--backend", "hf"] (rubric only, no vLLM).
ARGS = []
LOG_TAIL = []                     # the last lines of the latest command's output (the RUN cell reads them)
REAP_GRACE_S, WATCH_S = 15, 5     # kill leftovers this long after a command exits; sample its children this often


def kill_run(pgid, seen=None):
    \"\"\"Kill what a command left behind. nvidia-smi lists no processes inside Kaggle's container, so the handles
    are the command's process group (sp starts each command in its own session; vLLM's workers inherit the group
    and keep it when the script dies and they are re-parented) and the descendants psutil saw while it ran
    ({{pid: create time}}: a worker that left the group is still found, a re-used pid is not).\"\"\"
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    try:
        import psutil
    except ImportError:
        return
    for pid, t in dict(seen or {{}}).items():
        try:
            q = psutil.Process(pid)
            if q.create_time() == t:
                q.kill()
        except psutil.Error:
            pass


def gpu_mem_used():
    \"\"\"MiB in use on each GPU (this query works inside the container; the per-process one does not).\"\"\"
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.isdigit()]


def wait_gpus_free(limit_mib=1024, wait_s=120):
    t = time.time()
    while time.time() - t < wait_s and any(m > limit_mib for m in gpu_mem_used()):
        time.sleep(5)
    print("GPU memory in use (MiB):", gpu_mem_used(), flush=True)


def sp(*args):
    \"\"\"Run the script in a fresh interpreter and its own process group, streaming its output (the tail into
    LOG_TAIL); return its exit code. REAP_GRACE_S after it exits, whatever it left is killed (kill_run): an
    orphaned worker would hold GPU memory, and the output pipe and so this loop open; sp kills it again on the
    way out, and a later command is never touched.\"\"\"
    p = subprocess.Popen([sys.executable, "-u", SCRIPT, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, start_new_session=True)
    done, seen = threading.Event(), {{}}

    def watch():
        try:
            import psutil
            root = psutil.Process(p.pid)
        except Exception:
            root = None
        while p.poll() is None:
            try:
                for c in root.children(recursive=True) if root is not None else []:
                    seen.setdefault(c.pid, c.create_time())
            except Exception:
                pass
            time.sleep(WATCH_S)
        if not done.wait(REAP_GRACE_S):
            kill_run(p.pid, seen)

    threading.Thread(target=watch, daemon=True).start()
    tail = collections.deque(maxlen=300)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        return p.wait()
    finally:
        done.set()
        LOG_TAIL[:] = list(tail)
        kill_run(p.pid, seen)"""

PLAN = """\
# Data (downloaded with the HF_TOKEN secret), items, prompts and the time budget; CPU only.
print("plan exit", sp("plan", *ARGS))"""

RUN = """\
# Generation. Exit 0: all done; 75: stopped at the session deadline with work left (run a new version with this
# output attached); 77: a self-check stopped it; 78 or 2: a setup error. Those are not retried. Any other failure
# is retried at most twice, each retry adding the remedy for the failure it saw to the earlier ones (FAILURES):
# the engine's start hung (the init watchdog's exit 76) or NCCL failed -> --nccl-p2p-disable, then eager;
# CUDA out of memory -> --enforce-eager with less --gpu-mem, then fewer sequences; the KV cache too small for
# --max-model-len -> eager with more --gpu-mem, then a shorter --max-model-len; a Triton / prefix-prefill kernel
# error -> --no-prefix-caching; a download error -> the same again; anything else -> eager, less --gpu-mem.
import re
FAILURES = [  # kind, what marks it in the log (the latest match in the tail wins), remedies in turn
    ("nccl", r"NCCL error|nccl\\w*Error|ProcessGroupNCCL|collective operation timeout",
     [["--nccl-p2p-disable"], ["--enforce-eager", "--gpu-mem", "0.85"]]),
    ("oom", r"CUDA out of memory|OutOfMemoryError|out of memory",
     [["--enforce-eager", "--gpu-mem", "0.85"], ["--gpu-mem", "0.80", "--max-num-seqs", "32"]]),
    ("kv_cache", r"stored in KV cache|No available memory for the cache blocks",
     [["--enforce-eager", "--gpu-mem", "0.93"], ["--max-model-len", "5120"]]),
    ("triton", r"OutOfResources|out of resource|prefix_prefill|triton\\.runtime|TritonError|CompilationError",
     [["--no-prefix-caching"]]),
    ("network", r"ConnectionError|ReadTimeout|HTTPError|Temporary failure in name resolution|IncompleteRead",
     [[]]),
]
REMEDIES = {kind: rem for kind, _, rem in FAILURES}
REMEDIES["init_hang"] = REMEDIES["nccl"]
REMEDIES["other"] = [["--enforce-eager", "--gpu-mem", "0.85"]]


def failure_kind(rc, log):
    if rc == 76:
        return "init_hang"
    last = {kind: max((m.end() for m in re.finditer(pat, log, re.I)), default=-1) for kind, pat, _ in FAILURES}
    kind = max(last, key=last.get)
    return kind if last[kind] >= 0 else "other"


def merged(extra, add):
    \"\"\"The options `extra` with `add` on top (a later value of an option wins).\"\"\"
    opts, toks, i = {}, list(extra) + list(add), 0
    while i < len(toks):
        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else None
        opts[toks[i]] = val
        i += 1 if val is None else 2
    return [x for k, v in opts.items() for x in ([k] if v is None else [k, v])]


def next_remedy(kind, extra, tried):
    \"\"\"The current options plus the first remedy for this kind of failure not tried yet, or None.\"\"\"
    for rem in REMEDIES[kind]:
        if (kind, tuple(rem)) not in tried:
            tried.add((kind, tuple(rem)))
            return merged(extra, rem)
    return None


extra, tried = [], set()
for attempt in range(3):
    rc = sp("run", *ARGS, *extra)
    print("run exit", rc, extra, flush=True)
    if rc in (0, 75, 77, 78, 2):
        if rc not in (0, 75):
            print("not retried: see the log above and README.md")
        break
    kind = failure_kind(rc, "".join(LOG_TAIL))
    nxt = next_remedy(kind, extra, tried) if attempt < 2 else None
    if nxt is None:
        print(f"failure: {kind}; no retry left for it: see README.md")
        break
    print(f"failure: {kind}; retrying with {nxt}", flush=True)
    extra = nxt
    wait_gpus_free()"""

EXPORT = """\
# What the local side reads: strong_probe/<model>/export/ (copy its contents to data/features/kaggle/).
print("export exit", sp("export", *ARGS))
n = 0
for root, dirs, files in sorted(os.walk("/kaggle/working")):
    n += len(files) + len(dirs)
    if "/export" in root or root.endswith("strong_probe"):
        for f in sorted(files):
            print(os.path.join(root, f), os.path.getsize(os.path.join(root, f)))
print(f"/kaggle/working holds {n} files and directories; Kaggle saves at most 500"
      + (" -- WARNING: close to the cap, see README.md" if n > 400 else ""))"""

OUTRO = """\
## Дальше

Скачайте Output (вкладка Output или `kaggle kernels output <user>/paiec-strong-probe -p ./kaggle_out`) и положите содержимое `strong_probe/<модель>/export/` в `data/features/kaggle/` репозитория (экспорт коммита D с `entropy/` — в отдельную `data/features/kaggle_d/`, см. README «Коммит D»), весь Output — в `data/features/kaggle_raw/<run>/`, затем локально:
`python experiments/strong_llm_eval.py --stage check-schema` и остальные стадии (см. README.md); для `experiments/harness.py` признаки по одному файлу даёт `python kaggle/strong_probe/strong_probe.py split-harness --export-dir data/features/kaggle`. Признаки энтропии (`entropy_*`) `split-harness` не раскладывает: их читает только `strong_llm_eval.py --job entropy` по правилу `ENTROPY_RULE` (README, «Коммит D»)."""


def _pin(src, name):
    m = re.search(r'^%s = "([^"]+)"' % name, src, re.M)
    if not m:
        raise ValueError(f"{name} not found in strong_probe.py")
    return m.group(1)


def _lines(text):
    return text.splitlines(keepends=True)


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(text)}


def writefile_cell(src):
    return _code(f"%%writefile {TARGET}\n" + src)


def build(src):
    """The notebook (a dict) for the script's source text."""
    cells = [_md(INTRO), _code(CLOCK),
             _code(INSTALL.format(vllm=_pin(src, "VLLM_PIN"), transformers=_pin(src, "TRANSFORMERS_PIN"))),
             writefile_cell(src), _code(HELPERS.format(target=TARGET)), _code(PLAN), _code(RUN), _code(EXPORT),
             _md(OUTRO)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                                    "language": "python", "sourceType": "notebook", "dataSources": []}},
            "nbformat": 4, "nbformat_minor": 4}


def dumps(nb):
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def script_in(nb):
    """The script text carried by a notebook's %%writefile cell."""
    for c in nb["cells"]:
        text = "".join(c["source"])
        if c["cell_type"] == "code" and text.startswith(f"%%writefile {TARGET}\n"):
            return text[len(f"%%writefile {TARGET}\n"):]
    return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    with open(SCRIPT, encoding="utf-8") as fh:
        text = dumps(build(fh.read()))
    if "--check" in argv:
        with open(NOTEBOOK, encoding="utf-8") as fh:
            same = fh.read() == text
        print("notebook up to date" if same else "notebook is stale: run build_notebook.py")
        return 0 if same else 1
    with open(NOTEBOOK, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
