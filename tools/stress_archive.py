"""S3 of docs/plans/p2_final_plan.md: robustness and latency of the built archive.

    python tools/stress_archive.py                 # the full check, writes results/stress_archive.json
    python tools/stress_archive.py --quick --out /tmp/x.json   # a smoke test of this harness only

What is stressed is the archive itself, dist/paiec.zip (archive-3, sha256
4a882cc7...; anything else is rebuilt with tools/build_submission.py first and
checked again), loaded the way the organisers' validator loads it: the zip is
extracted to a fresh directory and model.py imported by the validator's own
_load_module (third_party/paiec_baseline/check_submission_zip.py, imported from
disk and not modified), and every output is judged by the validator's own
_assert_finite_probability. predict is called as the organisers' streaming
client calls it, predict(copy.deepcopy(input), copy.deepcopy(labeled))
(tools/streaming_ingestion.py), and as the validator does, predict(input,
labeled=...). The validator itself runs once in full, as tools/build_submission.py
runs it, and must print OK.

Parts (each in a fresh interpreter of its own, so its peak memory is its own):

  (i) fuzz. 10,000 predict calls on one loaded archive, deterministic from
      --seed: case i is drawn by random.Random seeded with the sha256 of
      (seed, i), its family from a fixed schedule (FAMILIES, shuffled the same
      way). Families: item_content (empty, whitespace, 1 MB ASCII, unicode,
      newline runs and hostile option patterns, 100k text, multiple choice A to
      J, 'Options:' blocks, drawing tags, control characters, non-strings);
      unicode in every field; item_features (None, numeric, nested dicts and
      lists to depth 100, JSON strings, 1 MB, malformed separators); non-empty
      interactors; odd release and access dates (non-ISO forms, impossible
      dates, non-strings, 100k strings); missing fields and broken input
      structure; labeled lists (None, malformed entries and labels, odd
      containers, duplicates, contradictions, 1 MB items, huge lists); all of
      these at once. Every case goes through a JSON round trip first, as the
      platform's transport does, except the 'raw' kinds (tuples, bytes,
      integer keys) that only an in-process caller could send. A huge labeled
      list keeps the platform's shape, the only constraint any case has: at
      most 31 labels a (subject, benchmark) pair, at most 700 pairs, 50
      benchmarks, 200 subjects a benchmark and 20,000 entries (the dense
      composition below has 140 pairs; the public data at most 82 subjects a
      benchmark). Shapes beyond it are the envelope probe, not gated.
      A sample of 500 cases is then replayed in reverse order in a fresh
      process (purity): a pure function of (input, labeled) gives the same bits.
  (ii) dense. 7 pseudo-benchmarks of paiec.testlike (DENSE: the catalogue's
      pseudo-benchmarks with disjoint items and at least 20 eligible pairs, one
      group of each parent first, researchcodebench's 100k-character items
      included once) x 20 subjects each (lowest stable_hash), split and acquired
      by paiec.official exactly as the platform does (sha256 random policy),
      at every checkpoint: labeled holds every pair's first B labels (4,340 at
      B31), and --targets-per-pair evaluation inputs of every pair are
      predicted, each call timed, by an archive loaded afresh for the
      checkpoint (a recreated worker). Both split scopes.
  (iii) parallel. The same checkpoints with --workers (16) processes at once,
      each a fresh interpreter that extracts and loads the archive itself and
      then gets labeled and its share of the targets as JSON through a pipe,
      as the platform's recreated workers do (targets dealt round-robin); the
      predictions should equal (ii)'s bit for bit (reported, not gated). Once
      more at B31 with every subject's release date in a non-ISO form, which
      sends paiec_rt.subjects to pandas in every worker.
  envelope (not gated). Single calls on labeled shapes the platform cannot
      send: thousands of distinct benchmark_ids, 3,000 subjects on one
      benchmark, 11,000 pairs of under two labels, long non-ISO release dates on
      many subjects. Recorded to show where the per-call cost grows.

The gate, as the plan reads: no exception escapes; every output is finite and
in (0, 1); p99 under 2 s and maximum under 30 s per call. The verdict reads it
on every predict call of (i), the purity replay, (ii) and (iii), pooled, the
one per-call criterion the plan states for the check; each part's own reading
is reported beside it (the fuzz part's p99 is set by the share of its huge-list
calls, each a fresh fit, so it describes the schedule more than any platform
call mix). A call's latency is its window: the fresh copy of input and labeled
inside the client's call window (copy.deepcopy) plus the predict call; the
predict call alone is reported beside it, and its CPU time, which the load of
other jobs on the machine does not inflate.

Nothing here writes item text: inputs from the public data reach worker
processes through pipes, and results hold counts, timings, digests and
pseudo-benchmark names only. BLAS and OpenMP run on one thread.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.abspath(__file__)
ARCHIVE = os.path.join(ROOT, "dist", "paiec.zip")
EXPECTED_SHA256 = "4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090"
VALIDATOR = os.path.join(ROOT, "third_party", "paiec_baseline", "check_submission_zip.py")
CLIENT = os.path.join(ROOT, "third_party", "paiec_baseline", "tools", "streaming_ingestion.py")
BUILD = os.path.join(ROOT, "tools", "build_submission.py")
OUT = os.path.join(ROOT, "results", "stress_archive.json")
PLAN = "docs/plans/p2_final_plan.md"
GATE = {"p99_s": 2.0, "max_s": 30.0}
BUDGETS = (0, 1, 3, 7, 15, 31)
SCOPES = ("pair", "benchmark")
DENSE = ("matharena::g5", "multi_swebench::g1", "multi_swebench::g3", "multi_swebench::g4",
         "multi_swebench::g5", "real_webagents::all", "researchcodebench::g0")
SUBJECTS = 20
#: research-package modules the dense composition is built with (not shipped)
LIB = ("paiec/data.py", "paiec/evaluator.py", "paiec/official.py", "paiec/testlike.py")
DATA_BENCHMARKS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
HEAVY = ("scipy", "sklearn", "pandas")
FAMILIES = (("content", 2000), ("unicode", 1000), ("features", 1500), ("interactors", 1000),
            ("dates", 1000), ("missing", 1500), ("labeled", 1500), ("combined", 500))
#: the platform's shape for a huge labeled list (module docstring)
HUGE = {"max_entries": 20_000, "max_pairs": 700, "max_benchmarks": 50,
        "max_subjects_per_benchmark": 200, "max_labels_per_pair": 31}
MB = 1 << 20


# --- small helpers --------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(path, ROOT)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def maxrss_mb(who=resource.RUSAGE_SELF):
    """Peak resident set size; ru_maxrss is bytes on macOS, KiB on Linux."""
    r = resource.getrusage(who).ru_maxrss
    return round(r / MB if sys.platform == "darwin" else r / 1024, 1)


def rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / MB, 1)
    except Exception:
        return None


def loadavg():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except Exception:
        return None


def rng_for(*parts):
    """A random.Random seeded from sha256 of the parts: the same stream in every
    process and every run (Python's hash() is salted per process)."""
    key = json.dumps([str(p) for p in parts]).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16], "big"))


def lat(xs):
    """p50, p90, p99 and max of a list of seconds."""
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def q(p):   # linear interpolation, numpy's default
        k = (len(s) - 1) * p
        f = math.floor(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {"n": len(s), "p50_s": round(q(0.50), 6), "p90_s": round(q(0.90), 6),
            "p99_s": round(q(0.99), 6), "max_s": round(s[-1], 6),
            "mean_s": round(sum(s) / len(s), 6),
            "over_2s": sum(x >= GATE["p99_s"] for x in s), "over_30s": sum(x >= GATE["max_s"] for x in s)}


# --- the archive, loaded as the validator loads it ------------------------------------

def validator():
    """The organisers' validator module, imported from disk unchanged."""
    spec = importlib.util.spec_from_file_location("check_submission_zip", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_archive(zip_path, workdir, V=None):
    """Extract the zip into workdir and import its model.py with the validator's
    _load_module, after dropping any earlier copy of the archive's modules, as
    a recreated worker starts without them. Returns (module, seconds)."""
    V = V or validator()
    for name in list(sys.modules):
        if name == "submission_model" or name.split(".")[0] == "paiec_rt":
            del sys.modules[name]
    t = time.perf_counter()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(workdir)
    module = V._load_module(Path(workdir) / "model.py", "submission_model", Path(workdir))
    return module, time.perf_counter() - t


def stray_modules(workdir):
    """The archive's modules (submission_model, paiec_rt) loaded from anywhere
    but the directory it was extracted to, and whether the research package
    paiec is loaded at all (only the processes that build the dense
    composition load it)."""
    where = os.path.realpath(workdir)
    stray = sorted(n for n, m in sys.modules.items()
                   if (n == "submission_model" or n.split(".")[0] == "paiec_rt")
                   and not os.path.realpath(getattr(m, "__file__", "") or "").startswith(where))
    return stray + (["(research package paiec loaded)"] if "paiec" in sys.modules else [])


def heavy_loaded():
    return sorted({n.split(".")[0] for n in sys.modules} & set(HEAVY))


def judge(V, value):
    """(validator accepts, finite, in the open interval (0, 1), native float, bits)."""
    try:
        V._assert_finite_probability(value, "predict()")
        accepted = True
    except Exception:
        accepted = False
    x = None
    if not isinstance(value, bool):
        try:
            x = float(value)
        except Exception:
            x = None
    finite = x is not None and math.isfinite(x)
    return (accepted, finite, finite and 0.0 < x < 1.0, type(value) is float,
            x.hex() if finite else "invalid:" + type(value).__name__)


def _stderr_kind(text):
    """The exception type a traceback printed to stderr ends with, if any."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    head = last.split(":", 1)[0].strip()
    return head if head.replace(".", "").replace("_", "").isalnum() and len(head) < 80 else "other"


def call(module, inp, lab, style="positional"):
    """One timed call as the client makes it: fresh deep copies of input and
    labeled, then predict. Never raises; returns a row of what happened."""
    pred = getattr(module, "PREDICTOR", None)
    before = getattr(pred, "failures", None)
    buf = io.StringIO()
    t0 = time.perf_counter()
    a = copy.deepcopy(inp)
    L = None if style == "none" else copy.deepcopy(lab)
    t1 = time.perf_counter()
    value, escaped = None, None
    c1 = time.process_time()
    try:
        with contextlib.redirect_stderr(buf):
            if style == "positional":
                value = module.predict(a, L)
            elif style == "keyword":
                value = module.predict(a, labeled=L)
            else:
                value = module.predict(a)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:            # noqa: B036 - SystemExit counts as escaping too
        escaped = type(exc).__name__
    t2 = time.perf_counter()
    c2 = time.process_time()
    try:
        mutated = bool(a != inp or (style != "none" and L != lab))
    except Exception:
        mutated = None
    after = getattr(pred, "failures", None)
    return {"value": value, "escaped": escaped, "predict_s": t2 - t1, "window_s": t2 - t0,
            "cpu_s": c2 - c1,
            "mutated": mutated, "fallback": (before is not None and after is not None and after > before),
            "stderr": _stderr_kind(buf.getvalue())}


class Tally:
    """Counts and latencies of a set of calls."""

    def __init__(self):
        self.n = 0
        self.c = Counter()
        self.window, self.predict, self.cpu = [], [], []
        self.stderr = Counter()
        self.escapes = Counter()

    def add(self, V, row):
        self.n += 1
        self.window.append(row["window_s"])
        self.predict.append(row["predict_s"])
        self.cpu.append(row["cpu_s"])
        if row["escaped"]:
            self.c["escaped"] += 1
            self.escapes[row["escaped"]] += 1
            return "escaped:" + row["escaped"]
        accepted, finite, open01, native, bits = judge(V, row["value"])
        self.c["validator_rejects"] += not accepted
        self.c["nonfinite"] += not finite
        self.c["outside_open_unit"] += not open01
        self.c["not_native_float"] += not native
        self.c["fallback"] += bool(row["fallback"])
        self.c["exactly_half"] += bool(finite and float(row["value"]) == 0.5)
        self.c["mutated_input"] += bool(row["mutated"])
        if row["stderr"]:
            self.stderr[row["stderr"]] += 1
        return bits

    def summary(self):
        keys = ("escaped", "validator_rejects", "nonfinite", "outside_open_unit", "not_native_float",
                "fallback", "exactly_half", "mutated_input")
        return {"calls": self.n, **{k: int(self.c[k]) for k in keys},
                "escaped_types": dict(self.escapes), "stderr_tracebacks": dict(self.stderr),
                "window": lat(self.window), "predict": lat(self.predict), "predict_cpu": lat(self.cpu)}

    def merge(self, other):
        self.n += other.n
        self.c.update(other.c)
        self.window += other.window
        self.predict += other.predict
        self.cpu += other.cpu
        self.stderr.update(other.stderr)
        self.escapes.update(other.escapes)


def tally_from(d):
    t = Tally()
    t.n = d["n"]
    t.c = Counter(d["c"])
    t.window, t.predict, t.cpu = list(d["window"]), list(d["predict"]), list(d.get("cpu", []))
    t.stderr, t.escapes = Counter(d["stderr"]), Counter(d["escapes"])
    return t


def tally_dump(t):
    return {"n": t.n, "c": dict(t.c), "window": t.window, "predict": t.predict, "cpu": t.cpu,
            "stderr": dict(t.stderr), "escapes": dict(t.escapes)}


# --- the fuzz cases -------------------------------------------------------------------

WORDS = ("alpha beta gamma delta value compute integer prime triangle function return server "
         "request agent browser click page repository patch test module error probability answer "
         "option correct choose following which what how many sum product matrix vector graph "
         "node edge string parse token compile runtime memory cache thread lock queue").split()
HARNESSES = ("", "default", "openhands", "swe-agent", "browser-use", "aider", "react", "codex-cli")
EFFORTS = ("", "low", "medium", "high", "xhigh", "minimal", "none")
LANGS = ("python", "java", "go", "rust", "c", "cpp", "javascript", "typescript")
ODD_DATES = (
    "", " ", "2024", "2024-05", "2024-5", "2024-05-13", "2024-5-3", "2024-05-13T12:00:00Z",
    "2024-05-13T12:00:00+05:30", "2024-05-13 12:00", "2024-05-13T12:00:00.123456789Z",
    "2024/05/13", "13/05/2024", "05/13/2024", "13.05.2024", "May 13, 2024", "13 May 2024",
    "May 2024", "Q2 2024", "2024 Q2", "mid-2024", "late 2023", "2024-W20", "2024-W20-1",
    "2024-135", "20240513", "1715558400", "1715558400000", "2024-13-01", "2024-02-30",
    "2024-00-00", "0000-00-00", "9999-12-31", "1677-09-21", "2262-04-12", "1066-10-14",
    "-2024-05-13", "+002024-05-13", "unknown", "N/A", "None", "null", "NaT", "nan", "TBD", "?",
    "２０２４-０５-１３", "2024年5月13日",
    "13 мая 2024", "  2024-05-13  ", "2024-05-13\n", "​2024-05-13",
    "2024-05-13\x00", "2024-05-13" + "0" * 50, "1" * 30, "2024-05-13T" + "1" * 1000,
    "yesterday", "now", "today", "2024-05-13/2024-06-01", "\ud800", "\U0001F600")
ODD_NONSTR = (2024, 2024.5, 20240513, 0, -1, None, True, False, [2024, 5, 13], {"y": 2024},
              float("nan"), float("inf"))
UNI_RANGES = ((0x1F600, 0x1F64F), (0x1D400, 0x1D4FF), (0x4E00, 0x4FFF), (0x0600, 0x06FF),
              (0x05D0, 0x05EA), (0x0400, 0x04FF), (0x0300, 0x036F), (0x200B, 0x200F),
              (0x202A, 0x202E), (0xD800, 0xDFFF), (0xE000, 0xE0FF), (0xFFF0, 0xFFFF),
              (0xFF01, 0xFF5E), (0x2028, 0x2029), (0x0000, 0x001F), (0x0080, 0x00FF))
HOSTILE = (
    lambda n: "\n" * n,
    lambda n: " \n" * (n // 2),
    lambda n: "\r\n" * (n // 2),
    lambda n: "\r" * n,
    lambda n: "(A) " * (n // 4),
    lambda n: "A. x\n" * (n // 5),
    lambda n: "A) \nB) \nC) \nD) \nE) \n" * (n // 20),
    lambda n: "A:" * (n // 2),
    lambda n: "A： " * (n // 3),
    lambda n: "\\begin{tikzpicture}" * (n // 19),
    lambda n: "[asy]" * (n // 5),
    lambda n: "\\begin{asy}(A) " * (n // 15),
    lambda n: "(A, B, C, D, " * (n // 13),
    lambda n: "(A, B, or C" * (n // 11),
    lambda n: "(" * n,
    lambda n: "A" * n,
    lambda n: " " * n,
    lambda n: "\t\x0b\x0c\x1c\x1d\x1e\x85\xa0" * (n // 8),
    lambda n: "\x00" * n,
    lambda n: "Options:\n" + "A. x\n" * (n // 5),
    lambda n: " " * n + "(A) x (B) y (C) z",
    lambda n: "\n" * (n // 2) + "A) x\nB) y\nC) z\n" + "\n" * (n // 2),
    lambda n: "=" * (n // 2) + ";" * (n // 2),
)


class Ctx:
    """Subject names and providers to draw from: the identity table and the
    provider list of the archive's own prior.json, so known subjects are hit."""

    def __init__(self, zip_path):
        try:
            with zipfile.ZipFile(zip_path) as z:
                d = json.loads(z.read("prior.json"))
            names = sorted(d["prior"]["table"])
            providers = list(d["prior"]["spec"]["providers"])
        except Exception:
            names, providers = [], []
        self.names = names + [f"model-{k}" for k in range(40)] + [n.title() for n in names[:30]]
        self.providers = providers + ["", "openai", "Unknown Lab", "  ", "Ⓐnthropic"]


def words(r, k):
    return " ".join(r.choice(WORDS) for _ in range(k))


def question(r):
    return words(r, r.randint(5, 40)).capitalize() + "?"


def iso(r):
    return f"{r.randint(2022, 2026)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"


def bench_id(r):
    return f"benchmark_{r.randint(100000, 999999)}"


def utext(r, k):
    out = []
    for _ in range(k):
        lo, hi = r.choice(UNI_RANGES)
        out.append(chr(r.randint(lo, hi)))
    return "".join(out)


def big(r, size, kind="ascii"):
    if kind == "unicode":
        chunk = utext(r, 2048)
    elif kind == "newlines":
        chunk = "".join(r.choice("\n\n\n \t\rA)(.:") for _ in range(4096))
    else:
        chunk = "".join(chr(r.randint(32, 126)) for _ in range(4096))
    return (chunk * (size // len(chunk) + 1))[:size]


def mcq(r, letters, header=""):
    style = r.choice(("{L}) ", "({L}) ", "{L}. ", "{L}: ", "{L}：", "{l}) ", "[{L}] "))
    opts = [style.format(L="ABCDEFGHIJ"[j], l="abcdefghij"[j]) + words(r, r.randint(1, 6))
            for j in range(letters)]
    sep = r.choice(("\n", "\n", " ", "  \n\n"))
    return question(r) + "\n" + header + sep.join(opts)


def subject(r, ctx):
    name = r.choice(ctx.names) if ctx.names else f"model-{r.randint(0, 99)}"
    return {"normalized_name": name, "provider": r.choice(ctx.providers or ["OpenAI"]),
            "release_date": iso(r), "access_date": iso(r), "harness": r.choice(HARNESSES),
            "reasoning_effort": r.choice(EFFORTS), "harness_version": r.choice(("", "1.0", "0.3.2", "v2")),
            "subject_features_extra": r.choice(("", "", "temperature=0", '{"tools": true}'))}


def features_ok(r):
    return r.choice(("", f"tier={r.choice(('near-term', 'medium', 'hard'))}",
                     f"competition=c{r.randint(0, 9)};problem_idx={r.randint(1, 30)}",
                     f"lang={r.choice(LANGS)}", f"website=w{r.randint(0, 6)};task_type=t{r.randint(0, 3)}",
                     f"paper=p{r.randint(0, 12)}"))


def item(r, bid, content=None):
    return {"item_content": question(r) if content is None else content,
            "item_features": features_ok(r), "interactors": "", "benchmark_id": bid}


def small_labeled(r, ctx, s, bid, n=None):
    """A labeled list shaped like a small run around the target: its own pair,
    a few other subjects and benchmarks, repeated items, 0/1 labels."""
    n = r.choice((0, 1, 2, 3, 5, 8, 13, 21, 31, 47, 62)) if n is None else n
    subjects = [s] + [subject(r, ctx) for _ in range(r.randint(0, 5))]
    bids = [bid] + [bench_id(r) for _ in range(r.randint(0, 3))]
    pools = {b: [item(r, b) for _ in range(r.randint(3, 40))] for b in bids}
    rates = [r.random() for _ in subjects]
    out = []
    for _ in range(n):
        j = r.randrange(len(subjects))
        b = r.choice(bids)
        out.append([[subjects[j], r.choice(pools[b])], int(r.random() < rates[j])])
    return out


def huge_labeled(r, ctx, s, bid, flat):
    """A huge list, 10,000 to 20,000 entries, in the platform's shape (HUGE):
    350 to 700 pairs of up to 31 labels on 5 to 50 benchmarks, at most 200
    subjects a benchmark, subjects shared between benchmarks. Structured gives
    every pair 15 or 31 distinct items (budget 15 or 31); 'flat' scatters the
    entries over the pairs at random, at most 31 a pair."""
    cap, per_max = HUGE["max_entries"], HUGE["max_labels_per_pair"]
    per = per_max if flat else r.choice((15, per_max))
    n_pairs = r.randint(max(350, -(-10_000 // per)), HUGE["max_pairs"])
    nb = r.randint(max(5, -(-n_pairs // HUGE["max_subjects_per_benchmark"])), HUGE["max_benchmarks"])
    ns = max(1, n_pairs // nb)
    people = [s] + [subject(r, ctx) for _ in range(r.randint(ns, max(ns, min(2 * ns, 400))))]
    bids = [bid] + [bench_id(r) for _ in range(nb - 1)]
    pairs = [(b, j) for b in bids for j in r.sample(range(len(people)), ns)]
    pools = {b: [item(r, b) for _ in range(r.randint(max(per, 40), 400))] for b in bids}
    rate = {p: r.random() for p in pairs}
    out = []
    if flat:
        top = min(cap, len(pairs) * per_max)
        n = r.randint(min(10_000, top), top)
        count = Counter()
        while len(out) < n:
            p = pairs[r.randrange(len(pairs))]
            if count[p] >= per_max:
                continue
            count[p] += 1
            out.append([[people[p[1]], r.choice(pools[p[0]])], int(r.random() < rate[p])])
    else:
        for p in pairs:
            for it in r.sample(pools[p[0]], per):
                out.append([[people[p[1]], it], int(r.random() < rate[p])])
        out = out[:cap]
    return out


def weird_content(r, kind=None):
    kinds = ("empty", "whitespace", "plain", "mcq_ae", "mcq_aj", "options_block", "mcq_inline",
             "listed", "code", "drawing", "long_100k", "big_1mb_ascii", "big_1mb_unicode",
             "big_1mb_newlines", "hostile_1mb", "control", "nonstr")
    kind = kind or r.choice(kinds)
    if kind == "empty":
        v = ""
    elif kind == "whitespace":
        v = r.choice((" ", "\n", "\t", " \n\t\r ", "　", "\xa0" * 10, "\n" * 5000))
    elif kind == "plain":
        v = question(r)
    elif kind == "mcq_ae":
        v = mcq(r, r.randint(2, 5))
    elif kind == "mcq_aj":
        v = mcq(r, r.randint(6, 10))
    elif kind == "options_block":
        v = mcq(r, r.randint(3, 10), header=r.choice(("Options:\n", "Choices:\n", "Answer choices: ",
                                                        "OPTIONS:", "Options: ")))
    elif kind == "mcq_inline":
        v = question(r) + " " + " ".join(f"({L}) {words(r, 2)}" for L in "ABCDEFGHIJ"[:r.randint(3, 10)])
    elif kind == "listed":
        v = question(r) + r.choice((" (A, B, C, D, or E)", " (A, B or C)", " (A,B,C,D,E,F,G, or H)",
                                     " (A, B, C, or E)", " (B, C, or D)"))
    elif kind == "code":
        v = "def f(A, B, C):\n    return (A) + (B)\n# A) not an option\nx = {'A': 1, 'B': 2}\n" * r.randint(1, 200)
    elif kind == "drawing":
        v = r.choice(("\\begin{tikzpicture}\\coordinate (A) at (0,0);\\coordinate (B) at (1,0);"
                      "\\coordinate (C) at (0,1);\\end{tikzpicture} Find AB.",
                      "[asy] label(\"(A)\"); label(\"(B)\"); label(\"(C)\"); [/asy] (A) 1 (B) 2",
                      "\\begin{tikzpicture} (A) (B) (C) (D) never closed",
                      "\\begin{picture}(A)\\end{picture}\\end{picture}" * 100))
    elif kind == "long_100k":
        v = (words(r, 50) + "\n") * (100_000 // 300)
    elif kind == "big_1mb_ascii":
        v = big(r, MB, "ascii")
    elif kind == "big_1mb_unicode":
        v = big(r, MB, "unicode")
    elif kind == "big_1mb_newlines":
        v = big(r, MB, "newlines")
    elif kind == "hostile_1mb":
        v = r.choice(HOSTILE)(MB)
    elif kind == "control":
        v = "".join(chr(r.randint(0, 31)) for _ in range(r.randint(1, 5000)))
    else:
        v = r.choice((0, 1, -1, 3.14, float("nan"), True, False, None, [], ["a", 1], {},
                      {"text": "q"}, [[["x"]]]))
    return kind, v


def nested(r, depth):
    v = r.choice((1, "x", None, 2.5, True, [], {}))
    for _ in range(depth):
        v = {"k" + str(r.randint(0, 3)): v, "n": r.randint(0, 9)} if r.random() < 0.6 else [v, "y"]
    return v


def weird_features(r, kind=None):
    kinds = ("none", "empty", "kv", "kv_many", "json_obj", "json_nested", "dict_obj", "dict_deep",
             "list_obj", "int", "float", "nan_inf", "bool", "long_5k", "big_1mb", "unicode_keys",
             "malformed", "brackets_deep", "numeric_values", "flags", "json_array_str", "json_scalar_str")
    kind = kind or r.choice(kinds)
    v = {"none": lambda: None, "empty": lambda: "", "kv": lambda: features_ok(r),
         "kv_many": lambda: ";".join(f"k{j}=v{r.randint(0, 5)}" for j in range(200)),
         "json_obj": lambda: json.dumps({"tier": r.choice(("a", "b")), "idx": r.randint(0, 9)}),
         "json_nested": lambda: json.dumps(nested(r, r.randint(5, 30))),
         "dict_obj": lambda: {"tier": r.choice(("a", "b")), "sub": {"x": [1, 2, {"y": None}]}},
         "dict_deep": lambda: nested(r, r.randint(30, 100)),
         "list_obj": lambda: ["tier=a", 1, None, ["x"]],
         "int": lambda: r.randint(-10 ** 12, 10 ** 12), "float": lambda: r.uniform(-1e9, 1e9),
         "nan_inf": lambda: r.choice((float("nan"), float("inf"), float("-inf"))),
         "bool": lambda: r.choice((True, False)),
         "long_5k": lambda: "tier=" + "x" * r.randint(4_001, 6_000),
         "big_1mb": lambda: big(r, MB, r.choice(("ascii", "unicode"))),
         "unicode_keys": lambda: ";".join(f"{utext(r, 3)}={utext(r, 4)}" for _ in range(5)),
         "malformed": lambda: r.choice(("=", ";;;", "a=b=c", "=v", "k=", ":", "a:b:c", "k=v;;k2",
                                         "[[[", '{"a":', "'unbalanced", '"q;=\\"', "{}", "{]",
                                         "k=v\nk=w\n", "\n\n\n", "a=[1;2];b={c;d}")),
         "brackets_deep": lambda: "[" * 10_000 + "k=v" + "]" * 5_000,
         "numeric_values": lambda: f"problem_idx={r.randint(1, 99)};score={r.random():.3f}",
         "flags": lambda: ";".join(r.sample(("hard", "timed", "visual", "multi", "long"), 3)),
         "json_array_str": lambda: "[1, 2, 3]", "json_scalar_str": lambda: "42"}[kind]()
    return kind, v


def weird_interactors(r, kind=None):
    kinds = ("kv", "json_list", "json_obj", "list_obj", "dict_obj", "plain", "unicode", "big_1mb",
             "none", "int", "nested", "many")
    kind = kind or r.choice(kinds)
    v = {"kv": lambda: f"user_simulator=gpt-4o;tools={r.choice(('browser', 'python', 'shell'))}",
         "json_list": lambda: json.dumps(["browser", "python", r.choice(("shell", "search"))]),
         "json_obj": lambda: json.dumps({"agent": "a" + str(r.randint(0, 9)), "turns": r.randint(1, 50)}),
         "list_obj": lambda: ["browser", "python"], "dict_obj": lambda: {"agent": "x", "n": 3},
         "plain": lambda: r.choice(("gpt-4o", "human", "simulated user", "env=docker")),
         "unicode": lambda: utext(r, 30), "big_1mb": lambda: big(r, MB),
         "none": lambda: None, "int": lambda: r.randint(0, 99), "nested": lambda: nested(r, 10),
         "many": lambda: ";".join(f"tool{j}=on" for j in range(100))}[kind]()
    return kind, v


def odd_date(r):
    u = r.random()
    if u < 0.75:
        return r.choice(ODD_DATES)
    if u < 0.95:
        return r.choice(ODD_NONSTR)
    return r.choice(("2024-05-13 " * 1000, "x" * 10_000, "May 13, 2024 " * 8_000))


def fuzz_schedule(seed, n):
    total = sum(w for _, w in FAMILIES)
    counts = [round(n * w / total) for _, w in FAMILIES]
    counts[-1] += n - sum(counts)
    sched = [f for (f, _), c in zip(FAMILIES, counts) for _ in range(c)]
    rng_for("paiec-s3-schedule", seed, n).shuffle(sched)
    return sched


LABELED_KINDS = ("none", "empty_list", "small", "medium", "huge_structured", "huge_flat",
                 "malformed_mix", "bad_labels", "odd_container", "odd_items", "big_items",
                 "all_ones", "all_zeros", "duplicates", "contradictory", "target_in_labeled",
                 "subject_elsewhere", "tuple_entries", "nested_extra", "many_groups",
                 "mcq_floor_labels", "long_text_items")
BAD_LABELS = ("1", "0", 0.5, None, True, False, 2, -1, float("nan"), float("inf"), 1.0, 0.0, [1],
              {"y": 1}, "", "true")


def weird_labeled(r, ctx, s, it, kind=None):
    """(kind, labeled, raw): raw lists go to predict without the JSON round trip."""
    bid = it.get("benchmark_id") if isinstance(it, dict) else bench_id(r)
    bid = bid if isinstance(bid, str) else bench_id(r)
    kind = kind or r.choice(LABELED_KINDS)
    raw = False
    if kind == "none":
        lab = None
    elif kind == "empty_list":
        lab = []
    elif kind == "small":
        lab = small_labeled(r, ctx, s, bid)
    elif kind == "medium":
        lab = small_labeled(r, ctx, s, bid, n=r.randint(200, 1_500))
    elif kind == "huge_structured":
        lab = huge_labeled(r, ctx, s, bid, flat=False)
    elif kind == "huge_flat":
        lab = huge_labeled(r, ctx, s, bid, flat=True)
    elif kind == "malformed_mix":
        lab = small_labeled(r, ctx, s, bid, n=r.randint(5, 80))
        bad = ([[s, it]], [s, it, 1], [[s], 1], [[s, it, it], 1], {"input": [s, it], "label": 1},
               "string", 5, None, [], [[None, it], 1], [[s, None], 1], [[s, "item"], 1],
               [[[s], [it]], 1], [[s, it], 1, "extra"], [["s", "i"], 0], [[{}, {}], 1])
        for _ in range(r.randint(1, len(lab) + 1)):
            lab.insert(r.randint(0, len(lab)), copy.deepcopy(r.choice(bad)))
    elif kind == "bad_labels":
        lab = small_labeled(r, ctx, s, bid, n=r.randint(5, 60))
        for e in lab:
            if r.random() < 0.5:
                e[1] = r.choice(BAD_LABELS)
    elif kind == "odd_container":
        lab = r.choice(({}, "", "[]", 0, 1, True, [[]], [None], {"labeled": []}, [[[], []]],
                        "[[[{}, {}], 1]]", 3.5))
    elif kind == "odd_items":
        lab = small_labeled(r, ctx, s, bid, n=r.randint(5, 60))
        for e in lab:
            if r.random() < 0.5:
                e[0][1] = dict(e[0][1])
                field = r.choice(("item_content", "item_features", "interactors", "benchmark_id"))
                e[0][1][field] = r.choice((None, 7, "", utext(r, 20), nested(r, 8), ["x"], 2.5))
    elif kind == "big_items":
        lab = small_labeled(r, ctx, s, bid, n=r.randint(3, 30))
        for e in lab[:3]:
            e[0][1] = dict(e[0][1], item_content=big(r, MB, r.choice(("ascii", "unicode"))))
    elif kind in ("all_ones", "all_zeros"):
        lab = [[[s, item(r, bid)], int(kind == "all_ones")] for _ in range(r.choice((1, 3, 7, 15, 31)))]
    elif kind == "duplicates":
        e = [[s, item(r, bid)], r.randint(0, 1)]
        lab = [e] * r.randint(2, 500) + small_labeled(r, ctx, s, bid, n=10)
    elif kind == "contradictory":
        x = item(r, bid)
        lab = [[[s, x], j % 2] for j in range(r.randint(2, 200))]
    elif kind == "target_in_labeled":
        lab = small_labeled(r, ctx, s, bid, n=r.randint(0, 30)) + [[[s, it], r.randint(0, 1)]] * r.randint(1, 3)
    elif kind == "subject_elsewhere":
        lab = [[[s, item(r, bench_id(r))], r.randint(0, 1)] for _ in range(r.randint(1, 93))]
    elif kind == "tuple_entries":
        raw = True
        lab = [((e[0][0], e[0][1]), e[1]) for e in small_labeled(r, ctx, s, bid, n=r.randint(1, 40))]
        lab = tuple(lab) if r.random() < 0.5 else lab
    elif kind == "nested_extra":
        lab = [[[s, it], 1, {"meta": nested(r, 5)}]] + small_labeled(r, ctx, s, bid, n=10)
    elif kind == "many_groups":
        lab = []
        for j in range(r.randint(100, 1_200)):
            x = item(r, bid)
            x["item_features"] = f"idx={j};grp=g{r.randint(0, 400)};lvl=l{r.randint(0, 3)}"
            lab.append([[s if r.random() < 0.3 else subject(r, ctx), x], r.randint(0, 1)])
    elif kind == "mcq_floor_labels":
        lab = [[[s, item(r, bid, content=mcq(r, r.randint(2, 10)))], int(r.random() < 0.2)]
               for _ in range(r.randint(1, 80))]
    else:   # long_text_items
        text = (words(r, 50) + "\n") * (100_000 // 300)
        lab = [[[s, item(r, bid, content=text + str(j))], r.randint(0, 1)] for j in range(r.randint(5, 60))]
    return kind, lab, raw


def fuzz_case(seed, i, family, ctx):
    """Case i: (kind, input, labeled, style, raw), drawn from its own stream."""
    r = rng_for("paiec-s3-fuzz", seed, i)
    s = subject(r, ctx)
    bid = bench_id(r)
    it = item(r, bid)
    lab, raw, kind = None, False, None
    style = r.choices(("positional", "keyword", "none"), (80, 15, 5))[0]
    inp = None
    if family == "content":
        kind, it["item_content"] = weird_content(r)
    elif family == "unicode":
        kind = r.choice(("content", "features", "interactors", "benchmark_id", "subject_name",
                         "provider", "labeled_items", "all_fields"))
        if kind in ("content", "all_fields"):
            it["item_content"] = utext(r, r.randint(1, 20_000))
        if kind in ("features", "all_fields"):
            it["item_features"] = f"{utext(r, 4)}={utext(r, 6)};lang={utext(r, 3)}"
        if kind in ("interactors", "all_fields"):
            it["interactors"] = utext(r, 50)
        if kind in ("benchmark_id", "all_fields"):
            it["benchmark_id"] = "benchmark_" + utext(r, 6)
        if kind in ("subject_name", "all_fields"):
            s["normalized_name"] = utext(r, 20)
        if kind in ("provider", "all_fields"):
            s["provider"] = utext(r, 10)
        lab = small_labeled(r, ctx, s, it["benchmark_id"])
        if kind in ("labeled_items", "all_fields"):
            for e in lab:
                e[0][1] = dict(e[0][1], item_content=utext(r, 200), item_features=f"{utext(r, 2)}=x")
    elif family == "features":
        kind, it["item_features"] = weird_features(r)
        lab = small_labeled(r, ctx, s, bid)
        for e in lab:
            if r.random() < 0.5:
                e[0][1] = dict(e[0][1], item_features=weird_features(r, kind)[1])
    elif family == "interactors":
        kind, it["interactors"] = weird_interactors(r)
        lab = small_labeled(r, ctx, s, bid)
        for e in lab:
            if r.random() < 0.5:
                e[0][1] = dict(e[0][1], interactors=weird_interactors(r)[1])
    elif family == "dates":
        kind = r.choice(("release_odd", "access_odd", "both_odd", "labeled_subjects_odd",
                         "access_before_release", "far_future", "mixed_in_labeled"))
        if kind in ("release_odd", "both_odd"):
            s["release_date"] = odd_date(r)
        if kind in ("access_odd", "both_odd"):
            s["access_date"] = odd_date(r)
        if kind == "access_before_release":
            s["release_date"], s["access_date"] = "2026-06-01", "2023-01-01"
        if kind == "far_future":
            s["release_date"] = r.choice(("2199-01-01", "9999-12-31", "3000"))
        lab = small_labeled(r, ctx, s, bid)
        if kind in ("labeled_subjects_odd", "mixed_in_labeled"):
            for e in lab:
                if r.random() < 0.7:
                    d = r.choice(ODD_DATES) if kind == "labeled_subjects_odd" else r.choice(
                        (iso(r), r.choice(ODD_DATES), r.choice(ODD_NONSTR)))
                    e[0][0] = dict(e[0][0], release_date=d)
    elif family == "missing":
        kinds = ("subject_missing", "item_missing", "subject_none_values", "item_none_values",
                 "subject_empty", "item_only_benchmark", "item_empty", "no_benchmark_id",
                 "extra_keys", "labeled_entries_missing", "input_tuple", "input_one", "input_three",
                 "input_empty", "input_none", "input_dict", "input_swapped", "subject_not_dict",
                 "item_not_dict", "input_str", "nested_input", "benchmark_id_types", "raw_types")
        kind = r.choice(kinds)
        if kind == "subject_missing":
            for k in r.sample(sorted(s), r.randint(1, len(s))):
                del s[k]
        elif kind == "item_missing":
            for k in r.sample(sorted(it), r.randint(1, len(it))):
                del it[k]
        elif kind == "subject_none_values":
            for k in r.sample(sorted(s), r.randint(1, len(s))):
                s[k] = None
        elif kind == "item_none_values":
            for k in r.sample(sorted(it), r.randint(1, len(it))):
                it[k] = None
        elif kind == "subject_empty":
            s = {}
        elif kind == "item_only_benchmark":
            it = {"benchmark_id": bid}
        elif kind == "item_empty":
            it = {}
        elif kind == "no_benchmark_id":
            del it["benchmark_id"]
        elif kind == "extra_keys":
            s["_sid"], s["subject_id"], it["item_id"], it["label"] = "x", "subject_1", "i1", 1
            it.update({f"extra{j}": j for j in range(20)})
        elif kind == "input_tuple":
            inp, raw = (s, it), True
        elif kind == "input_one":
            inp = [s]
        elif kind == "input_three":
            inp = [s, it, {"extra": 1}]
        elif kind == "input_empty":
            inp = []
        elif kind == "input_none":
            inp = None
        elif kind == "input_dict":
            inp = {"subject": s, "item": it}
        elif kind == "input_swapped":
            inp = [it, s]
        elif kind == "subject_not_dict":
            inp = [r.choice(("subject", 1, None, ["a"], 2.5)), it]
        elif kind == "item_not_dict":
            inp = [s, r.choice(("item", 1, None, ["a"], 2.5))]
        elif kind == "input_str":
            inp = r.choice(("", "[{}, {}]", "input"))
        elif kind == "nested_input":
            inp = [[s, it]]
        elif kind == "benchmark_id_types":
            it["benchmark_id"] = r.choice((7, None, "", "b" * 10_000, ["b"], {"b": 1}, 1.5, True))
        else:   # raw_types: what only an in-process caller sends
            raw = True
            it["item_content"] = r.choice((b"bytes content", bytearray(b"ab"), ("t", "u")))
            it["item_features"] = r.choice(({1: "a", 2: "b"}, ("k", "v"), b"k=v"))
        lab = small_labeled(r, ctx, s if isinstance(s, dict) else {}, bid)
        if kind == "labeled_entries_missing":
            for e in lab:
                if r.random() < 0.5:
                    e[0][0] = {k: v for k, v in e[0][0].items() if r.random() < 0.5}
                    e[0][1] = {k: v for k, v in e[0][1].items() if r.random() < 0.5}
    elif family == "labeled":
        kind, lab, raw = weird_labeled(r, ctx, s, it)
        style = "positional" if r.random() < 0.85 else "keyword"
    else:   # combined
        kind = "combined"
        if r.random() < 0.35:
            it["item_content"] = weird_content(r)[1]
        if r.random() < 0.35:
            it["item_features"] = weird_features(r)[1]
        if r.random() < 0.35:
            it["interactors"] = weird_interactors(r)[1]
        if r.random() < 0.35:
            s["release_date"] = odd_date(r)
        if r.random() < 0.35:
            for k in r.sample(sorted(s), r.randint(1, 4)):
                del s[k]
        if r.random() < 0.2:
            it["benchmark_id"] = r.choice((None, 3, "", utext(r, 5)))
        if r.random() < 0.35:
            s["normalized_name"] = utext(r, 12)
        lk, lab, raw = weird_labeled(r, ctx, s, it)
        kind = "combined:" + lk
    if inp is None:
        inp = [s, it]
    if not raw:     # the platform's JSON transport
        inp, lab = json.loads(json.dumps([inp, lab]))
    return kind, inp, lab, style, raw


# --- part (i): fuzz, and its replay ---------------------------------------------------

def _fuzz_child(zip_path, seed, n, q):
    V = validator()
    ctx = Ctx(zip_path)
    tmp = tempfile.mkdtemp(prefix="paiec-s3-fuzz-")
    out = {"loadavg_start": loadavg(), "rss_before_load_mb": rss_mb()}
    try:
        module, load_s = load_archive(zip_path, tmp, V)
        out.update(load_s=round(load_s, 4), model=getattr(module, "MODEL", None),
                   rss_after_load_mb=rss_mb())
        sched = fuzz_schedule(seed, n)
        total, fam, kinds = Tally(), defaultdict(Tally), defaultdict(Tally)
        bits, slow, sizes = [], [], Counter()
        t_start = time.perf_counter()
        for i, family in enumerate(sched):
            kind, inp, lab, style, raw = fuzz_case(seed, i, family, ctx)
            row = call(module, inp, lab, style)
            b = total.add(V, row)
            fam[family].add(V, row)
            kinds[f"{family}/{kind.split(':')[0] if family != 'combined' else kind}"].add(V, row)
            bits.append(b)
            slow.append((row["window_s"], i, family, kind, len(lab) if isinstance(lab, (list, tuple)) else None))
            if isinstance(lab, (list, tuple)) and len(lab) >= 10_000:
                sizes["huge_calls"] += 1
                sizes["huge_max_entries"] = max(sizes["huge_max_entries"], len(lab))
            if (i + 1) % 500 == 0:
                print(f"  fuzz {i + 1}/{n}: {time.perf_counter() - t_start:.0f}s, "
                      f"escaped {total.c['escaped']}, invalid {total.c['outside_open_unit']}", flush=True)
        slow.sort(reverse=True)
        out.update(
            calls=n, wall_s=round(time.perf_counter() - t_start, 1), tally=tally_dump(total),
            families={f: t.summary() for f, t in sorted(fam.items())},
            kinds={k: t.summary() for k, t in sorted(kinds.items())},
            slowest=[{"index": i, "family": f, "kind": k, "labeled_entries": le, "window_s": round(w, 4)}
                     for w, i, f, k, le in slow[:15]],
            huge=dict(sizes), bits=bits,
            failures=getattr(getattr(module, "PREDICTOR", None), "failures", None),
            unconverged=getattr(getattr(module, "PREDICTOR", None), "unconverged", None),
            stray_modules=stray_modules(tmp), heavy_loaded=heavy_loaded())
    except BaseException as exc:            # noqa: B036
        out["harness_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.update(maxrss_mb=maxrss_mb(), loadavg_end=loadavg())
    q.put(out)


def _replay_child(zip_path, seed, n, indices, q):
    """The cases at `indices`, in that order, on an archive loaded afresh."""
    V = validator()
    ctx = Ctx(zip_path)
    sched = fuzz_schedule(seed, n)
    tmp = tempfile.mkdtemp(prefix="paiec-s3-replay-")
    out = {}
    try:
        module, _ = load_archive(zip_path, tmp, V)
        t, bits = Tally(), {}
        for i in indices:
            kind, inp, lab, style, raw = fuzz_case(seed, i, sched[i], ctx)
            bits[i] = t.add(V, call(module, inp, lab, style))
        out.update(tally=tally_dump(t), bits=bits)
    except BaseException as exc:            # noqa: B036
        out["harness_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out["maxrss_mb"] = maxrss_mb()
    q.put(out)


# --- parts (ii) and (iii): the dense composition ---------------------------------------

def composition(seed):
    """The run: DENSE pseudo-benchmarks x SUBJECTS subjects, and what it holds."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import warnings
    warnings.simplefilter("ignore")
    from paiec import data as D
    from paiec import testlike as T
    from paiec.evaluator import stable_hash
    pairs = D.load_pairs()
    cat = T.build_catalogue(pairs)
    by = {p.name: p for p in cat.pseudos}
    run, info = [], {}
    for name in DENSE:
        ps = sorted(by[name].pairs, key=lambda p: stable_hash(seed, "s3-dense", name, p.subject_id))
        if len(ps) < SUBJECTS:
            raise RuntimeError(f"{name} has {len(ps)} eligible pairs, fewer than {SUBJECTS}")
        chosen = ps[:SUBJECTS]
        chars = [len(str(r.item.get("item_content") or "")) for p in chosen for r in p.responses]
        info[name] = {"eligible_pairs": len(ps), "subjects": SUBJECTS, "kind": by[name].kind,
                      "items": len(by[name].items),
                      "mean_item_chars": round(sum(chars) / max(1, len(chars)))}
        run += chosen
    return run, info


def checkpoints(run, seed, scope, per_pair):
    """(B, labeled, targets) for every checkpoint: paiec.official's split, sha256
    acquisition and shared labeled, and the first `per_pair` distinct
    evaluation inputs of every pair (the platform predicts each distinct input
    once)."""
    from paiec import official as O
    slots = O._slots(run, seed, scope)
    hook = O.make_hook(None, False)
    rt = O._Runtime(None, False, None)
    targets, distinct = [], 0
    for s in slots:
        seen, mine = set(), []
        for inp, _, key in s.targets:
            if key not in seen:
                seen.add(key)
                mine.append(inp)
        distinct += len(mine)
        targets += mine[:per_pair]
    for B in BUDGETS:
        if B:
            O._acquire(slots, B, hook, rt)
        labeled = [e for s in slots for e in s.acquired[:B]]
        per = [min(B, len(s.acquired)) for s in slots]
        yield B, labeled, targets, {"distinct_eval_inputs": distinct, "pairs": len(slots),
                                    "labels_per_pair_min": min(per), "labels_per_pair_max": max(per)}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _dense_child(zip_path, seed, per_pair, scopes, q):
    """(ii): one archive loaded afresh per checkpoint, every target predicted
    and timed in turn, as one recreated worker serving them all."""
    V = validator()
    out = {"loadavg_start": loadavg(), "scopes": {}}
    try:
        t0 = time.perf_counter()
        run, info = composition(seed)
        out.update(composition=info, build_s=round(time.perf_counter() - t0, 1))
        for scope in scopes:
            res = {}
            for B, labeled, targets, meta in checkpoints(run, seed, scope, per_pair):
                tmp = tempfile.mkdtemp(prefix="paiec-s3-dense-")
                try:
                    module, load_s = load_archive(zip_path, tmp, V)
                    t, bits, first = Tally(), [], None
                    tb = time.perf_counter()
                    for j, inp in enumerate(targets):
                        row = call(module, inp, labeled, "positional")
                        bits.append(t.add(V, row))
                        if j == 0:
                            first = row["window_s"]
                    pred = getattr(module, "PREDICTOR", None)
                    res[str(B)] = {
                        "labels": len(labeled), "targets": len(targets), **meta,
                        "labeled_json_mb": round(len(json.dumps(labeled)) / MB, 2),
                        "load_s": round(load_s, 4), "first_call_s": round(first, 4),
                        "wall_s": round(time.perf_counter() - tb, 2), "tally": tally_dump(t),
                        "fit_fallbacks": getattr(pred, "failures", None),
                        "unconverged": getattr(pred, "unconverged", None),
                        "bits_digest": digest(bits), "bits": bits, "rss_mb": rss_mb(),
                        "stray_modules": stray_modules(tmp), "loadavg": loadavg(),
                        "composition_digest": digest([labeled, targets])}
                    print(f"  dense {scope} B{B}: {len(labeled)} labels, {len(targets)} calls, "
                          f"first {first:.3f}s, p99 {lat(t.window)['p99_s']:.3f}s, max {max(t.window):.3f}s, "
                          f"wall {time.perf_counter() - tb:.0f}s", flush=True)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
            out["scopes"][scope] = res
    except BaseException as exc:            # noqa: B036
        out["harness_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out.update(maxrss_mb=maxrss_mb(), loadavg_end=loadavg())
    q.put(out)


def _worker(zip_path, w, conn, q):
    """(iii): one recreated evaluation worker: a fresh interpreter that extracts
    and loads the archive, then reads labeled and its targets as JSON from a
    pipe and predicts each in turn."""
    t_wall = time.time()
    out = {"w": w, "t_start": t_wall}
    tmp = tempfile.mkdtemp(prefix=f"paiec-s3-w{w}-")
    try:
        t0 = time.perf_counter()
        V = validator()
        module, load_s = load_archive(zip_path, tmp, V)
        out["load_s"] = load_s
        raw_l, raw_t = conn.recv_bytes(), conn.recv_bytes()
        conn.close()
        labeled, targets = json.loads(raw_l), json.loads(raw_t)
        del raw_l, raw_t
        out["setup_s"] = time.perf_counter() - t0
        t, bits = Tally(), []
        first = None
        for j, inp in enumerate(targets):
            row = call(module, inp, labeled, "positional")
            bits.append(t.add(V, row))
            if j == 0:
                first = row["window_s"]
        pred = getattr(module, "PREDICTOR", None)
        out.update(tally=tally_dump(t), bits=bits, first_call_s=first,
                   fit_fallbacks=getattr(pred, "failures", None),
                   unconverged=getattr(pred, "unconverged", None),
                   stray_modules=stray_modules(tmp), heavy_loaded=heavy_loaded())
    except BaseException as exc:            # noqa: B036
        out["harness_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.update(t_end=time.time(), maxrss_mb=maxrss_mb())
    q.put(out)


def _sample_rss(pids, stop, peak):
    """Peak of the summed RSS of the live workers, sampled every 50 ms."""
    try:
        import psutil
    except Exception:
        return
    while not stop.is_set():
        total = 0
        for pid in pids:
            try:
                total += psutil.Process(pid).memory_info().rss
            except Exception:
                pass
        peak[0] = max(peak[0], total)
        stop.wait(0.05)


def _parallel(ctx, zip_path, labeled, targets, workers):
    """One checkpoint served by `workers` fresh processes at once, targets dealt
    round-robin as paiec.official deals them."""
    lab_bytes = json.dumps(labeled).encode()
    shards = [list(range(w, len(targets), workers)) for w in range(workers)]
    q = ctx.Queue()
    procs, sends = [], []
    for w, idx in enumerate(shards):
        if idx:
            mine, theirs = ctx.Pipe()
            procs.append(ctx.Process(target=_worker, args=(zip_path, w, theirs, q)))
            sends.append((mine, json.dumps([targets[j] for j in idx]).encode()))
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    stop, peak = threading.Event(), [0]
    sampler = threading.Thread(target=_sample_rss, args=([p.pid for p in procs], stop, peak), daemon=True)
    sampler.start()
    for (mine, tg), p in zip(sends, procs):
        try:
            mine.send_bytes(lab_bytes)
            mine.send_bytes(tg)
        except (OSError, EOFError):
            pass                    # the worker died; collect() and the tally count it
        mine.close()
    got = collect(q, procs, len(procs))
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    stop.set()
    sampler.join()
    by_w = {r["w"]: r for r in got}
    bits = [None] * len(targets)
    lost = 0
    for w, idx in enumerate(shards):
        r = by_w.get(w)
        if idx and (r is None or "tally" not in r):
            lost += len(idx)        # a worker that died: its every call counts as escaping
            continue
        for j, b in zip(idx, r.get("bits", []) if r else []):
            bits[j] = b
    got.sort(key=lambda r: r["w"])
    starts, ends = [r["t_start"] for r in got], [r["t_end"] for r in got]
    return got, bits, {"wall_s": round(wall, 2), "workers": len(procs), "workers_answered": len(got),
                       "calls_lost_with_a_worker": lost,
                       "all_running_together_s": round(max(0.0, min(ends) - max(starts)), 2) if got else 0.0,
                       "peak_total_rss_mb": round(peak[0] / MB, 1) if peak[0] else None,
                       "exitcodes": sorted(Counter(p.exitcode for p in procs).items())}


def odd_dates(obj):
    """Every subject's ISO release date as '13 May 2024', which paiec_rt.subjects
    hands to pandas."""
    def fix(s):
        d = s.get("release_date")
        try:
            s = dict(s, release_date=datetime.date.fromisoformat(d[:10]).strftime("%d %B %Y"))
        except Exception:
            pass
        return s

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):     # an input
        return [fix(obj[0]), obj[1]]
    return [[[fix(e[0][0]), e[0][1]], e[1]] for e in obj]


def _parallel_child(zip_path, seed, per_pair, scopes, workers, q):
    """(iii): the driver. Builds the same composition and serves every
    checkpoint with `workers` fresh processes."""
    ctx = mp.get_context("spawn")
    out = {"loadavg_start": loadavg(), "scopes": {}}
    try:
        run, info = composition(seed)
        for scope in scopes:
            res = {}
            for B, labeled, targets, meta in checkpoints(run, seed, scope, per_pair):
                got, bits, m = _parallel(ctx, zip_path, labeled, targets, workers)
                res[str(B)] = summarize_workers(got, bits, m, labeled, targets)
                res[str(B)]["composition_digest"] = digest([labeled, targets])
                res[str(B)]["loadavg"] = loadavg()
                print(f"  parallel {scope} B{B}: {workers} workers, wall {m['wall_s']:.1f}s, "
                      f"p99 {res[str(B)]['calls']['window']['p99_s']:.3f}s, "
                      f"max {res[str(B)]['calls']['window']['max_s']:.3f}s, "
                      f"peak RSS {m['peak_total_rss_mb']} MB", flush=True)
                if scope == "pair" and B == BUDGETS[-1]:
                    lab2 = odd_dates(labeled)
                    tg2 = [odd_dates(t) for t in targets]
                    got2, bits2, m2 = _parallel(ctx, zip_path, lab2, tg2, workers)
                    out["odd_dates_B31"] = summarize_workers(got2, bits2, m2, lab2, tg2)
                    out["odd_dates_B31"]["loadavg"] = loadavg()
                    print(f"  parallel odd dates B{B}: wall {m2['wall_s']:.1f}s, first calls max "
                          f"{out['odd_dates_B31']['first_call']['max_s']:.3f}s, heavy "
                          f"{out['odd_dates_B31']['heavy_loaded']}", flush=True)
            out["scopes"][scope] = res
    except BaseException as exc:            # noqa: B036
        out["harness_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out.update(maxrss_mb=maxrss_mb(), loadavg_end=loadavg())
    q.put(out)


def summarize_workers(got, bits, m, labeled, targets):
    t = Tally()
    for r in got:
        if "tally" in r:
            t.merge(tally_from(r["tally"]))
    if m["calls_lost_with_a_worker"]:
        t.n += m["calls_lost_with_a_worker"]
        t.c["escaped"] += m["calls_lost_with_a_worker"]
        t.escapes["worker_died"] += m["calls_lost_with_a_worker"]
    return {"labels": len(labeled), "targets": len(targets), **m, "tally": tally_dump(t),
            "calls": t.summary(), "bits": bits, "bits_digest": digest(bits),
            "harness_errors": [r["harness_error"] for r in got if "harness_error" in r],
            "archive_load": lat([r["load_s"] for r in got if "load_s" in r]),
            "setup": lat([r["setup_s"] for r in got if "setup_s" in r]),
            "first_call": lat([r["first_call_s"] for r in got if r.get("first_call_s") is not None]),
            "fit_fallbacks": sum(r.get("fit_fallbacks") or 0 for r in got),
            "unconverged": sum(r.get("unconverged") or 0 for r in got),
            "worker_maxrss_mb": {"max": max((r["maxrss_mb"] for r in got), default=None),
                                 "sum": round(sum(r["maxrss_mb"] for r in got), 1)},
            "stray_modules": sorted({x for r in got for x in r.get("stray_modules", [])}),
            "heavy_loaded": sorted({x for r in got for x in r.get("heavy_loaded", [])})}


# --- envelope probe (not gated) ---------------------------------------------------------

ENVELOPE = (("singleton_benchmarks_2000", 2_000, "singletons"),
            ("singleton_benchmarks_20000", 20_000, "singletons"),
            ("one_benchmark_3000_subjects_x5", 3_000, "wide"),
            ("flat_20000_over_50x500", 20_000, "flat"),
            ("long_nonISO_dates_on_5_subjects", 5, "dates"))


def _envelope_child(zip_path, seed, q):
    V = validator()
    out = {"loadavg_start": loadavg(), "probes": {}}
    tmp = tempfile.mkdtemp(prefix="paiec-s3-env-")
    try:
        module, _ = load_archive(zip_path, tmp, V)
        r = rng_for("paiec-s3-envelope", seed)

        def subj(j, date=None):
            return {"normalized_name": f"model-{j}", "provider": "OpenAI",
                    "release_date": date or f"2025-{1 + j % 12:02d}-01", "access_date": "2026-01-01",
                    "harness": "h", "reasoning_effort": "", "harness_version": "",
                    "subject_features_extra": ""}

        def it(b, k):
            return {"item_content": f"Q {b}-{k}: " + words(r, 8), "item_features": f"tier={k % 5}",
                    "interactors": "", "benchmark_id": f"benchmark_{100000 + b}"}

        for name, n, shape in ENVELOPE:
            if shape == "singletons":
                lab = [[[subj(j), it(j, j)], int(r.random() < 0.4)] for j in range(n)]
            elif shape == "wide":
                lab = [[[subj(j), it(0, r.randrange(300))], int(r.random() < 0.4)]
                       for j in range(n) for _ in range(5)]
            elif shape == "flat":
                lab = [[[subj(r.randrange(500)), it(r.randrange(50), r.randrange(300))],
                        int(r.random() < 0.4)] for _ in range(n)]
            else:
                lab = [[[subj(j, ("May 13, 2024 " * 80_000)[:MB - j]), it(0, k)], int(r.random() < 0.4)]
                       for j in range(n) for k in range(5)]
            t = Tally()
            rows = [call(module, [subj(10 ** 6 + c), it(c % 7, 10 ** 6 + c)], lab) for c in range(3)]
            for row in rows:
                t.add(V, row)
            out["probes"][name] = {"labeled_entries": len(lab), "first_call_s": round(rows[0]["window_s"], 3),
                                   "later_calls_s": [round(x["window_s"], 4) for x in rows[1:]],
                                   **{k: v for k, v in t.summary().items() if k not in ("window", "predict")}}
            print(f"  envelope {name}: first {rows[0]['window_s']:.2f}s", flush=True)
    except BaseException as exc:            # noqa: B036
        out["harness_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.update(maxrss_mb=maxrss_mb(), loadavg_end=loadavg())
    q.put(out)


# --- driver ----------------------------------------------------------------------------

def collect(q, procs, n):
    """n results from q; stops early, with what it has, once every process has
    exited and the queue stays empty (a process that died never answers)."""
    import queue as _queue
    got = []
    while len(got) < n:
        try:
            got.append(q.get(timeout=5))
        except _queue.Empty:
            if not any(p.is_alive() for p in procs):
                try:
                    while len(got) < n:
                        got.append(q.get(timeout=1))
                except _queue.Empty:
                    break
    return got


def child(target, *args):
    """Run target(*args, q) in a fresh spawned interpreter; its result dict."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q))
    t = time.perf_counter()
    p.start()
    got = collect(q, [p], 1)
    p.join()
    got = got[0] if got else {"harness_error": f"process died, exit code {p.exitcode}"}
    got["child_wall_s"] = round(time.perf_counter() - t, 1)
    got["child_exitcode"] = p.exitcode
    return got


def ensure_archive(path, rebuild):
    """sha256 of the archive; rebuilt once with tools/build_submission.py when it
    is missing or not archive-3."""
    have = sha256_file(path) if os.path.isfile(path) else None
    rebuilt = None
    if have != EXPECTED_SHA256 and rebuild:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([sys.executable, BUILD], cwd=ROOT, env=env, capture_output=True, text=True)
        rebuilt = {"returncode": r.returncode, "tail": (r.stdout + r.stderr).strip().splitlines()[-3:]}
        have = sha256_file(path) if os.path.isfile(path) else None
    return have, rebuilt


def run_validator(path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, VALIDATOR, os.path.abspath(path)], cwd=tmp, env=env,
                           capture_output=True, text=True)
    text = (r.stdout + r.stderr).strip()
    return {"returncode": r.returncode, "ok": r.returncode == 0 and r.stdout.startswith("OK"),
            "output": text.replace(os.path.abspath(path), rel(path))}


def strip_bits(d):
    """The result without per-call bits (kept only as digests)."""
    if isinstance(d, dict):
        return {k: strip_bits(v) for k, v in d.items() if k not in ("bits", "tally")}
    if isinstance(d, list):
        return [strip_bits(x) for x in d]
    return d


def gate_of(t):
    s = t.summary()
    ok_exc = s["escaped"] == 0
    ok_val = (s["validator_rejects"] == 0 and s["nonfinite"] == 0 and s["outside_open_unit"] == 0)
    ok_p99 = s["calls"] > 0 and s["window"]["p99_s"] < GATE["p99_s"]
    ok_max = s["calls"] > 0 and s["window"]["max_s"] < GATE["max_s"]
    return {"calls": s["calls"], "no_exception_escapes": ok_exc, "finite_and_in_open_unit": ok_val,
            "p99_under_2s": ok_p99, "max_under_30s": ok_max,
            "p99_window_s": s["window"].get("p99_s"), "max_window_s": s["window"].get("max_s"),
            "p99_predict_s": s["predict"].get("p99_s"), "max_predict_s": s["predict"].get("max_s"),
            "p99_predict_cpu_s": s["predict_cpu"].get("p99_s"), "max_predict_cpu_s": s["predict_cpu"].get("max_s"),
            "p99_predict_under_2s": s["calls"] > 0 and s["predict"]["p99_s"] < GATE["p99_s"],
            "max_predict_under_30s": s["calls"] > 0 and s["predict"]["max_s"] < GATE["max_s"],
            "pass": ok_exc and ok_val and ok_p99 and ok_max}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fuzz", type=int, default=10_000)
    ap.add_argument("--replay", type=int, default=500, help="fuzz cases replayed in a fresh process")
    ap.add_argument("--targets-per-pair", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scopes", default=",".join(SCOPES))
    ap.add_argument("--no-envelope", action="store_true")
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--quick", action="store_true", help="a smoke test of this harness, not the check")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    if args.quick:
        args.fuzz, args.replay, args.targets_per_pair, args.workers = 400, 40, 2, 4
        args.scopes, args.no_envelope = "pair", True
        if os.path.abspath(args.out) == OUT:
            sys.exit("--quick writes elsewhere: pass --out")
    scopes = tuple(s for s in args.scopes.split(",") if s)
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.perf_counter()

    sha, rebuilt = ensure_archive(args.archive, not args.no_rebuild)
    if sha != EXPECTED_SHA256:
        sys.exit(f"the archive's sha256 is {sha}, not archive-3's {EXPECTED_SHA256}; nothing stressed")
    with zipfile.ZipFile(args.archive) as z:
        members = {i.filename: i.file_size for i in z.infolist()}
    print(f"archive {rel(args.archive)}: sha256 {sha} (archive-3), {len(members)} members", flush=True)
    val = run_validator(args.archive)
    print(f"validator: {val['output']}", flush=True)

    print("part (i) fuzz", flush=True)
    fuzz = child(_fuzz_child, args.archive, args.seed, args.fuzz)
    sched = fuzz_schedule(args.seed, args.fuzz)
    idx = sorted(rng_for("paiec-s3-replay", args.seed).sample(range(args.fuzz), min(args.replay, args.fuzz)),
                 reverse=True)
    print(f"purity replay of {len(idx)} cases, reverse order, fresh process", flush=True)
    replay = child(_replay_child, args.archive, args.seed, args.fuzz, idx)
    mism = [i for i in idx if "bits" in fuzz and "bits" in replay and replay["bits"].get(i) != fuzz["bits"][i]]
    print("part (ii) dense", flush=True)
    dense = child(_dense_child, args.archive, args.seed, args.targets_per_pair, scopes)
    print("part (iii) parallel", flush=True)
    par = child(_parallel_child, args.archive, args.seed, args.targets_per_pair, scopes, args.workers)
    env = None
    if not args.no_envelope:
        print("envelope probe (not gated)", flush=True)
        env = child(_envelope_child, args.archive, args.seed)

    # pooled tallies and the gate
    errors = {k: d.get("harness_error") for k, d in (("fuzz", fuzz), ("replay", replay), ("dense", dense),
                                                      ("parallel", par), ("envelope", env or {}))}
    t_fuzz = tally_from(fuzz["tally"]) if "tally" in fuzz else Tally()
    t_replay = tally_from(replay["tally"]) if "tally" in replay else Tally()
    t_dense, t_par = Tally(), Tally()
    same_bits, comp_same = {}, {}
    for scope in scopes:
        for B in map(str, BUDGETS):
            d = dense.get("scopes", {}).get(scope, {}).get(B)
            p = par.get("scopes", {}).get(scope, {}).get(B)
            if d:
                t_dense.merge(tally_from(d["tally"]))
            if p:
                t_par.merge(tally_from(p["tally"]))
            if d and p:
                same_bits[f"{scope}/B{B}"] = d["bits"] == p["bits"]
                comp_same[f"{scope}/B{B}"] = d["composition_digest"] == p["composition_digest"]
    t_odd = tally_from(par["odd_dates_B31"]["tally"]) if "odd_dates_B31" in par else Tally()
    t_par_all = Tally()
    t_par_all.merge(t_par)
    t_par_all.merge(t_odd)
    pooled = Tally()
    for t in (t_fuzz, t_replay, t_dense, t_par_all):
        pooled.merge(t)
    parts = {"fuzz": gate_of(t_fuzz), "replay": gate_of(t_replay), "dense": gate_of(t_dense),
             "parallel": gate_of(t_par_all)}
    pooled_gate = gate_of(pooled)
    harness_ok = (not any(v for k, v in errors.items() if k != "envelope")
                  and all(v for v in comp_same.values()) and bool(comp_same))
    # the plan states one per-call criterion for the check: read on every call, pooled; each
    # part's own reading is reported beside it (the fuzz part's p99 is set by the share of its
    # huge-list calls, each a fresh fit, which no platform call mix fixes)
    verdict = ("HARNESS ERROR" if not harness_ok else "PASS" if pooled_gate["pass"] else "FAIL")

    # a full dense run, every distinct evaluation input, at the measured serial rate
    projection = {}
    for scope in scopes:
        tot = 0.0
        for B in map(str, BUDGETS):
            d = dense.get("scopes", {}).get(scope, {}).get(B)
            if d:
                tot += d["distinct_eval_inputs"] * tally_from(d["tally"]).summary()["window"]["mean_s"]
        projection[scope] = {"serial_s": round(tot, 1), f"over_{args.workers}_workers_s":
                             round(tot / args.workers, 1), "platform_limit_s": 8 * 3600}

    inputs = {rel(p): sha256_file(p) for p in (args.archive, VALIDATOR, CLIENT, *[os.path.join(ROOT, x) for x in LIB])
              if os.path.isfile(p)}
    for b in DATA_BENCHMARKS:
        for f in ("response.parquet", "items.parquet", "subjects.parquet"):
            p = os.path.join(ROOT, "data", b, f)
            if os.path.isfile(p):
                inputs[rel(p)] = sha256_file(p)
    out = {
        "check": "S3", "plan": PLAN,
        "what": "robustness and latency of archive-3 through the validator's loader: fuzz, dense "
                "composition at every checkpoint, parallel recreated workers",
        "gate": {"text": "No exception escapes; every output is finite and in (0, 1); p99 under 2 s and "
                         "maximum under 30 s per call.",
                 "read_on": "every predict call of the fuzz, its replay, the dense composition and the "
                            "parallel workers, pooled (the verdict); each part alone reported beside it; "
                            "latency is wall-clock of the call window (fresh deep copies of input and "
                            "labeled, then predict), with predict alone and its CPU time beside it",
                 "verdict_on": "pooled",
                 "pooled": pooled_gate, "parts_alone": parts, "harness_ok": harness_ok, "verdict": verdict},
        "archive": {"path": rel(args.archive), "sha256": sha, "is_archive_3": sha == EXPECTED_SHA256,
                    "rebuilt": rebuilt, "members": members},
        "validator": val,
        "fuzz": {**strip_bits({k: v for k, v in fuzz.items() if k not in ("bits",)}),
                 "summary": t_fuzz.summary(), "schedule": dict(Counter(sched)), "huge_shape": HUGE,
                 "families_listed": [f for f, _ in FAMILIES]},
        "replay": {"cases": len(idx), "order": "reverse", "fresh_process": True,
                   "bit_mismatches": len(mism), "mismatched_indices": mism[:50],
                   "summary": t_replay.summary(), "maxrss_mb": replay.get("maxrss_mb"),
                   "harness_error": replay.get("harness_error")},
        "dense": {**strip_bits(dense), "summary": t_dense.summary()},
        "parallel": {**strip_bits(par), "summary": t_par.summary(),
                     "odd_dates_summary": t_odd.summary() if t_odd.n else None,
                     "bits_equal_to_dense": same_bits, "composition_equal_to_dense": comp_same},
        "projection_full_dense_run": projection,
        "envelope_not_gated": strip_bits(env) if env else None,
        "memory_peak_mb": {"fuzz_process": fuzz.get("maxrss_mb"), "replay_process": replay.get("maxrss_mb"),
                           "dense_process": dense.get("maxrss_mb"), "parallel_driver": par.get("maxrss_mb"),
                           "parallel_worker_max": max((v["worker_maxrss_mb"]["max"] for s in par.get("scopes", {}).values()
                                                       for v in s.values()), default=None),
                           "parallel_concurrent_total_max": max((v.get("peak_total_rss_mb") or 0
                                                                 for s in par.get("scopes", {}).values()
                                                                 for v in s.values()), default=None),
                           "envelope_process": (env or {}).get("maxrss_mb")},
        "harness_errors": errors,
        "machine": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "machine": platform.machine(), "cpus": os.cpu_count(),
                    "loadavg_start": fuzz.get("loadavg_start"), "loadavg_end": loadavg(),
                    "threads": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "params": {"seed": args.seed, "fuzz": args.fuzz, "replay": args.replay,
                   "targets_per_pair": args.targets_per_pair, "workers": args.workers, "scopes": list(scopes),
                   "dense": list(DENSE), "subjects": SUBJECTS, "budgets": list(BUDGETS), "quick": args.quick},
        "provenance": {"script": rel(SCRIPT), "script_sha256": sha256_file(SCRIPT), "git_head": git_head(),
                       "inputs": inputs, "command": "python " + " ".join([rel(SCRIPT)] + sys.argv[1:]),
                       "date_utc": started.isoformat(timespec="seconds"),
                       "wall_s": round(time.perf_counter() - t_all, 1)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False, default=str)
        f.write("\n")
    s = pooled.summary()
    print(f"pooled: {s['calls']} calls, escaped {s['escaped']}, invalid {s['validator_rejects']}, "
          f"outside (0,1) {s['outside_open_unit']}, window p50 {s['window']['p50_s']:.4f}s "
          f"p99 {s['window']['p99_s']:.4f}s max {s['window']['max_s']:.3f}s", flush=True)
    print(f"purity replay mismatches {len(mism)}; dense = parallel bits: {same_bits}", flush=True)
    print(f"gate verdict: {verdict}  -> {rel(args.out)}", flush=True)


if __name__ == "__main__":
    main()
