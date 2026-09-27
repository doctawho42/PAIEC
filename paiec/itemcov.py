"""Item covariates with a known sign, read off one item dict: no labels, no model.

Every function here takes the item exactly as predict() receives it (item_content,
item_features, interactors, benchmark_id; all strings) and returns a number, a
class name or None. None means the item carries no such cue. Nothing is fitted,
nothing depends on the benchmark's name, and nothing raises: a malformed item
gives None (or the neutral class). The sign of each covariate is fixed in
advance, higher = harder, from the literature and the item's own semantics,
never from the public labels; experiments/itemcov_eval.py checks the sign leave
one benchmark out and scores each covariate through experiments/harness.py.

Candidates (docs/findings.md, "Item covariates with a known sign"):

  ordinal_difficulty  an item_features key whose name says difficulty (difficulty,
            level, tier, stars, rating, elo, hardness, complexity, grade) and
            whose value is a number or a word on easy < medium < hard, or a
            count whose name says work (n_steps, num_hops, n_files, ...).
            hier.select_keys drops every all-numeric key, and learns word
            levels only as zero-mean group offsets from labels; this reads
            them as ordinals with a known direction. Absent from the five
            public benchmarks, so only the parser is tested on them.
  position  a numeric index key of a problem within its source (problem_idx,
            question_number, task_index, ...): later problems of a contest are
            usually harder. matharena's problem_idx is the one public case
            (median within-competition Spearman with Rasch difficulty 0.23,
            positive in 60% of competitions; rethink2 methodology-critic
            f_ordinal).
  answer_format  proof / mcq / code / short / free, read off item_content;
            format_score orders them mcq -1 < short = code = free 0 < proof +1
            (a proof graded by judges is harder than a checkable short answer,
            which is harder than picking among k options). Code and free text
            are not ordered against short answers and get 0. n_options cuts
            TikZ / Asymptote drawings out before paiec.mcq reads options (their
            point labels "(A) at (0,0)" are not options) and also reads an
            "(A, B, C, D, or E)" list.
  log_length  log(1 + characters of item_content).
  stated_size  the amount of work the item itself states ("Approximately 7
            line(s) of code", "about 5 steps"): researchcodebench states the
            lines of every TODO block, whose gold size correlates 0.53 with
            Rasch difficulty there (rethink2 attempt-signals a4). The first
            statement counts (a later one is usually a worked example).
  image_ref  1 if the text points at an image it does not contain ("See
            image", a markdown image, an <image> tag), else 0.

covariates(item) returns all of them as floats (None where absent), which is
what a harness covariate map is built from.

Measured (docs/findings.md): no candidate passes the sign rule (log_length,
the only one present on all five units, changes sign) or the harness gate;
stated_size is the one with item signal (Spearman +0.50 on researchcodebench,
-0.0022 ALC on its test-like pairs with a forced per-pair slope from B7) and
exists on one public benchmark only. Research-only: nothing here ships.
"""
from __future__ import annotations

import math
import re

try:                                    # the same parser hier groups items with
    from .hier import parse_features as _parse_features
except Exception:                       # pragma: no cover - hier always imports in the repo
    _parse_features = None
try:
    from .mcq import n_options as _n_options
except Exception:                       # pragma: no cover
    _n_options = None

#: key tokens that name a difficulty ordinal (higher value = harder)
DIFFICULTY_TOKENS = frozenset({"difficulty", "difficult", "difficulties", "hardness", "hard",
                               "level", "lvl", "tier", "complexity", "stars", "star", "rating",
                               "elo", "grade"})
#: key tokens that name a count of required work (more = harder)
WORK_TOKENS = frozenset({"steps", "step", "hops", "hop", "files", "subtasks", "subgoals",
                         "actions", "turns", "constraints", "operations", "ops"})
COUNT_PREFIXES = frozenset({"n", "num", "number", "count", "nb", "total"})
#: a key with one of these tokens is not a difficulty whatever else it says
#: (repository stars, a log level, a user rating, a zoom level ...)
EXCLUDE_TOKENS = frozenset({"repo", "repository", "github", "user", "users", "log", "logging",
                            "zoom", "api", "reading", "popularity", "review", "reviews",
                            "content", "age", "noise", "compression", "verbosity", "debug"})
#: key tokens of a position within a source
POSITION_TOKENS = frozenset({"idx", "index", "position", "pos", "number", "num", "no", "nr",
                             "order", "rank", "id"})
POSITION_SUBJECTS = frozenset({"problem", "question", "task", "item", "exercise", "q", "p"})

#: word values of a difficulty key on one scale (easy < medium < hard)
WORDS = {
    "trivial": -2.0, "very easy": -2.0, "very_easy": -2.0, "veryeasy": -2.0, "elementary": -2.0,
    "easy": -1.0, "simple": -1.0, "basic": -1.0, "beginner": -1.0, "low": -1.0,
    "introductory": -1.0, "intro": -1.0, "novice": -1.0,
    "easy-medium": -0.5, "easy_medium": -0.5, "easy/medium": -0.5,
    "medium": 0.0, "moderate": 0.0, "intermediate": 0.0, "normal": 0.0, "mid": 0.0,
    "average": 0.0, "standard": 0.0, "med": 0.0,
    "medium-hard": 0.5, "medium_hard": 0.5, "medium/hard": 0.5,
    "hard": 1.0, "difficult": 1.0, "advanced": 1.0, "high": 1.0, "challenging": 1.0,
    "very hard": 2.0, "very_hard": 2.0, "veryhard": 2.0, "expert": 2.0, "extreme": 2.0,
    "insane": 2.0, "very difficult": 2.0,
}
FORMAT_SCORE = {"mcq": -1.0, "short": 0.0, "code": 0.0, "free": 0.0, "proof": 1.0}

_NUMBER = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")
_ORDINAL_NUMBER = re.compile(r"^(?:(?:level|lvl|l|tier|t|grade|g|stage|rank|difficulty|d)\s*[-_:#]?\s*)?"
                             r"([-+]?\d+(?:\.\d+)?)\s*(?:/\s*\d+(?:\.\d+)?|stars?|out of \d+)?$", re.I)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
#: an instruction to write a proof: an imperative at a sentence start, or the
#: grader's words ("write a proof", "your proof"); "we prove that" in a paper is not one
_PROOF = re.compile(r"(?:^|[.!?:;]\s+|\n\s*|\$\s+)(?:[Pp]rove|[Ss]how|[Dd]emonstrate)\s+that\b"
                    r"|\b(?:write|give|provide|present)\s+(?:a|an|the)\s+(?:complete\s+|full\s+|rigorous\s+|"
                    r"detailed\s+)?proof\b|\b(?:a|the) (?:rigorous|complete|full|formal) proof\b"
                    r"|\bproof solution\b|\byour proof\b|\bability to prove\b", re.M)
#: drawing code, whose labelled points "(A) at (0,0)" are not answer options
_DRAWING = re.compile(r"\\begin\{(tikzpicture|asy|picture)\}.*?(\\end\{\1\}|$)", re.S)
_MCQ_WORDS = re.compile(r"multiple[- ]choice|which of the following|choose (one|the correct|the best)"
                        r"|answer (options|choices)\b", re.I)
_SHORT = re.compile(r"\\boxed|final answer|answer is an? (integer|number|real|positive)|numerical answer"
                    r"|\bhow many\b|\bcompute\b|\bfind the (value|number|sum|product|remainder|smallest"
                    r"|largest|maximum|minimum)\b", re.I)
#: code in the text: a fence, a traceback, a TODO, or a definition at a line start
#: (asymptote's "import graph;" and "defined over" are not code)
_CODE = re.compile(r"```|\bTraceback \(most recent call last\)|\bTODO\b"
                   r"|^[ \t]*(?:def [A-Za-z_]\w*\s*\(|class [A-Za-z_]\w*\s*[:(]|import [A-Za-z_][\w.]*[ \t]*$"
                   r"|from [\w.]+ import \w|#include\s*[<\"]|(?:public|private|protected) (?:static )?\w+"
                   r"|func \w+\(|fn \w+\(|function \w+\()", re.M)
_SIZE_STRICT = re.compile(r"\b(?:approximately|about|around|roughly|exactly|at most|at least|~)\s*(\d{1,5})\s+"
                          r"(?:line\(s\)|lines?|steps?|sub-?tasks?|actions?|hops?)\b", re.I)
_IMAGE = re.compile(r"\bsee (the )?(image|figure|picture|diagram)\b|!\[[^\]]*\]\(|<image\b|\[image\]"
                    r"|<img\b", re.I)
TEXT_CHARS = 400_000                    # longer contents are read on their first and last half
PROOF_CHARS = 3_000                     # where a proof or options instruction is looked for


def _s(v) -> str:
    try:
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        return v if isinstance(v, str) else str(v)
    except Exception:
        return ""


def _content(item) -> str:
    try:
        t = _s(item.get("item_content")) if isinstance(item, dict) else ""
    except Exception:
        return ""
    if len(t) > TEXT_CHARS:
        t = t[:TEXT_CHARS // 2] + "\n" + t[-TEXT_CHARS // 2:]
    return t


def features(item) -> dict:
    """key -> value string of item_features and interactors ('interactors.' keys),
    parsed as hier parses them; {} when there are none or they are unreadable."""
    if not isinstance(item, dict) or _parse_features is None:
        return {}
    try:
        out = dict(_parse_features(item.get("item_features")))
        for k, v in _parse_features(item.get("interactors")).items():
            out["interactors." + str(k)] = v
        return {str(k): _s(v) for k, v in out.items()}
    except Exception:
        return {}


def key_tokens(key) -> list:
    """Lower-case tokens of a key: split at non-alphanumerics and camelCase."""
    try:
        k = _CAMEL.sub("_", _s(key))
        return [t for t in re.split(r"[^a-z0-9]+", k.lower()) if t]
    except Exception:
        return []


def key_kind(key) -> str | None:
    """'difficulty', 'work', 'position' or None for one item_features key."""
    toks = key_tokens(key)
    if not toks or any(t in EXCLUDE_TOKENS for t in toks):
        return None
    ts = set(toks)
    if ts & DIFFICULTY_TOKENS:
        return "difficulty"
    if ts & WORK_TOKENS and (ts & COUNT_PREFIXES or len(toks) == 1 or toks[-1] in WORK_TOKENS):
        return "work"
    if ts & POSITION_TOKENS and (ts & POSITION_SUBJECTS or toks in (["idx"], ["index"], ["position"])):
        if toks[-1] == "id" and "idx" not in ts:          # problem_id names an item, not a place
            return None
        return "position"
    return None


def ordinal_value(value) -> float | None:
    """A difficulty value on its own scale, higher = harder: a number (possibly
    inside a string such as 'Level 3', 'L2', '3/5', '3 stars'), a run of star
    characters, or a word on easy < medium < hard (WORDS). None otherwise."""
    try:
        v = _s(value).strip()
        if not v or len(v) > 64:
            return None
        low = v.lower().strip(" .\"'")
        if low in WORDS:
            return WORDS[low]
        stars = sum(1 for ch in v if ch in "★⭐*")
        if stars and stars == len(v.replace(" ", "")):
            return float(stars)
        norm = re.sub(r"[\s_]+", " ", low)
        if norm in WORDS:
            return WORDS[norm]
        m = _ORDINAL_NUMBER.match(norm)
        if m:
            x = float(m.group(1))
            return x if math.isfinite(x) else None
        for w in sorted(WORDS, key=len, reverse=True):         # 'very hard problem'
            if re.search(r"\b" + re.escape(w) + r"\b", norm):
                return WORDS[w]
        return None
    except Exception:
        return None


def _keyed(item, kind) -> float | None:
    try:
        f = features(item)
        vals = []
        for k in sorted(f):
            if key_kind(k) == kind:
                x = ordinal_value(f[k]) if kind == "difficulty" else _number(f[k])
                if x is not None:
                    vals.append(x)
        return vals[0] if vals else None
    except Exception:
        return None


def _number(v) -> float | None:
    try:
        s = _s(v).strip()
        if _NUMBER.fullmatch(s):
            x = float(s)
            return x if math.isfinite(x) else None
        return None
    except Exception:
        return None


def ordinal_difficulty(item) -> float | None:
    """The first difficulty-named key's value (ordinal_value), else the first
    work-count key's value, sorted by key name; None when the item has neither."""
    x = _keyed(item, "difficulty")
    if x is None:
        x = _keyed(item, "work")
        if x is not None and x >= 0:
            x = math.log1p(x)
    return x


def position(item) -> float | None:
    """log(1 + the problem's index within its source) from a position-named
    numeric key (problem_idx, question_number, ...); None when absent."""
    x = _keyed(item, "position")
    if x is None or x < 0:
        return None
    return math.log1p(x)


def n_options(item) -> int:
    """Answer options (paiec.mcq.n_options, with TikZ / Asymptote drawings cut
    out first, or an '(A, B, C, D, or E)' list), 0 when there are none."""
    try:
        t = _DRAWING.sub(" ", _content(item))
        k = _n_options(t) if _n_options is not None else 0
        if k:
            return int(k)
        m = re.search(r"\(\s*A\s*,\s*B\s*,\s*(?:C\s*,\s*)?(?:D\s*,\s*)?(?:E\s*,\s*)?or\s+([C-H])\s*\)", t)
        return "ABCDEFGH".index(m.group(1)) + 1 if m else 0
    except Exception:
        return 0


def answer_format(item) -> str:
    """'proof', 'mcq', 'code', 'short' or 'free', the first that matches in
    that order: an instruction to write a proof in the first PROOF_CHARS
    characters; answer options (paiec.mcq's, or '(A, B, C, D, or E)', or the
    words); code in the text (a fence, a traceback, a TODO, a definition at a
    line start); a short checkable answer (boxed, 'final answer', 'how many',
    ...); anything else."""
    try:
        t = _content(item)
        if not t.strip():
            return "free"
        if _PROOF.search(t[:PROOF_CHARS]):
            return "proof"
        if n_options(item) >= 3 or _MCQ_WORDS.search(t[:PROOF_CHARS]):
            return "mcq"
        if _CODE.search(t):
            return "code"
        if _SHORT.search(t):
            return "short"
        return "free"
    except Exception:
        return "free"


def format_score(item) -> float:
    return FORMAT_SCORE.get(answer_format(item), 0.0)


def log_length(item) -> float:
    try:
        return math.log1p(len(_s(item.get("item_content")) if isinstance(item, dict) else ""))
    except Exception:
        return 0.0


def stated_size(item) -> float | None:
    """log(1 + the first amount of work the text states: 'Approximately 7
    line(s) of code', 'about 5 steps'); None when it states none."""
    try:
        m = _SIZE_STRICT.search(_content(item))
        if not m:
            return None
        x = float(m.group(1))
        return math.log1p(x) if math.isfinite(x) and x >= 0 else None
    except Exception:
        return None


def image_ref(item) -> float:
    try:
        return 1.0 if _IMAGE.search(_content(item)) else 0.0
    except Exception:
        return 0.0


#: name -> function; the harness covariates, higher = harder for every one
COVARIATES = {
    "ordinal_difficulty": ordinal_difficulty,
    "position": position,
    "format_score": format_score,
    "log_length": log_length,
    "stated_size": stated_size,
    "image_ref": image_ref,
}


def covariates(item) -> dict:
    """Every covariate of one item: name -> float, or None where the item lacks
    the cue. Never raises."""
    out = {}
    for name, fn in COVARIATES.items():
        try:
            x = fn(item)
            out[name] = None if x is None else float(x)
        except Exception:
            out[name] = None
    return out
