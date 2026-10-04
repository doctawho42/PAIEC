"""Build the technical report PDF from docs/report/draft.md.

    python tools/build_report_pdf.py                  # dist/report/paiec_report.pdf
    python tools/build_report_pdf.py --render auto    # also PNGs of a few pages
    python tools/build_report_pdf.py --check-deterministic
    python tools/build_report_pdf.py --strict         # fail on overfull boxes, missing
                                                      # glyphs, broken links, placeholders

Pipeline: the figures named by the draft (docs/report/fig/<name>.png) are copied into
a build directory, converted from their SVGs to vector PDF when rsvg-convert or
cairosvg is available, else used as the committed 200-dpi PNGs; pandoc reads the
draft as GitHub Markdown with tools/report_pdf/defaults.yaml (template.tex and the
report.lua filter, which changes layout only, never a word or a number); XeLaTeX runs
until cross-references settle. Figures carry the draft's own captions, which
docs/report/figures.md says are condensed from its full ones; the build checks the
draft's figure labels, files and sections against figures.md's placement table
(--captions full prints figures.md's full captions instead, for review only: they
change the text and some no longer fit a page). Everything is written under
dist/report/ (gitignored): the PDF, build/ (the .tex, .log and figures) and build.json
(inputs with sha256, tool versions, page counts, warnings, every overfull box, and a
check that every number and word of the draft is in the PDF's text, and that the text
has no semicolon turned into U+037E). The draft's TeX math ($...$, $$...$$) is set by
unicode-math; the text check reads it as printed: TeX command names and the arguments
that print nothing are dropped, the numbers and the words of \text{...} and the like are
kept, and both sides are folded by NFKC (mathematical italic letters, sub- and
superscript digits) with U+2212 read as a hyphen. The title block's "Keywords:" line
fills the PDF's keywords metadata and is not printed; CITATION.cff repeats it.

Needs pandoc (>= 3.1) and XeLaTeX with the packages template.tex loads (TeX Live has
them all; the build asks kpsewhich first and names any that are missing); pdftotext
(poppler) for the text check, skipped without it, and pdftoppm for --render. Looks for
them on PATH, then in /opt/homebrew/bin, /opt/local/bin, /usr/local/bin and
/Library/TeX/texbin; PANDOC, XELATEX, RSVG_CONVERT, PDFTOTEXT, PDFTOPPM and KPSEWHICH
override. No network.

Deterministic as far as LaTeX allows: SOURCE_DATE_EPOCH (from the environment, else
the commit time of HEAD) fixes the PDF's dates and ID, so two builds of the same
inputs with the same tools are byte-identical (--check-deterministic builds twice
and compares).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "docs" / "report" / "draft.md"
FIGURES_MD = ROOT / "docs" / "report" / "figures.md"
KIT = ROOT / "tools" / "report_pdf"
OUT = ROOT / "dist" / "report"
PDF_NAME = "paiec_report.pdf"
MAX_PDF_BYTES = 50_000_000      # OpenReview takes one PDF of at most 50 MB
OVERFULL_LIMIT_PT = 3.0         # --strict fails above this
TOOL_DIRS = ("/opt/homebrew/bin", "/opt/local/bin", "/usr/local/bin", "/Library/TeX/texbin")
# Placeholders --strict refuses in the title block. The repository one is written in two
# pieces so that a search for it finds only the places to fill, not this check.
PLACEHOLDERS = ("TODO", "OWNER" + "/REPO")
MAX_LATEX_RUNS = 5


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------- tools

def find_tool(name: str, env_var: str | None = None) -> str | None:
    """The path of an executable: $env_var, else PATH, else the usual install dirs."""
    if env_var and os.environ.get(env_var):
        p = os.environ[env_var]
        return p if Path(p).exists() else None
    p = shutil.which(name)
    if p:
        return p
    for d in TOOL_DIRS:
        q = Path(d) / name
        if q.exists() and os.access(q, os.X_OK):
            return str(q)
    return None


def tool_version(path: str) -> str:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def latex_packages() -> list[str]:
    """The packages template.tex loads, in order."""
    text = (KIT / "template.tex").read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"^\s*\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", text, re.M):
        out += [p.strip() for p in m.group(1).split(",") if p.strip()]
    return list(dict.fromkeys(out))


def missing_latex_packages(xelatex: str) -> list[str] | None:
    """The template's packages kpsewhich cannot find; None if there is no kpsewhich."""
    kpse = Path(xelatex).with_name("kpsewhich")
    kpse = str(kpse) if kpse.exists() else find_tool("kpsewhich", "KPSEWHICH")
    if not kpse:
        return None
    pkgs = latex_packages()
    try:
        p = subprocess.run([kpse, *[f"{x}.sty" for x in pkgs]], capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    found = {Path(line.strip()).stem for line in p.stdout.splitlines() if line.strip()}
    return [x for x in pkgs if x not in found]


def rel(path: Path) -> str:
    """A path relative to the repository if it is inside it, else as given."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ dates and git

def git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def source_date_epoch(environ: dict | None = None) -> int:
    """SOURCE_DATE_EPOCH from the environment, else HEAD's commit time, else 0."""
    environ = os.environ if environ is None else environ
    v = environ.get("SOURCE_DATE_EPOCH", "").strip()
    if v:
        if not v.isdigit():
            raise BuildError(f"SOURCE_DATE_EPOCH must be an integer, got {v!r}")
        return int(v)
    ct = git("log", "-1", "--format=%ct", "HEAD")
    return int(ct) if ct and ct.isdigit() else 0


# ----------------------------------------------------------------- the title block

# The labelled lines of the title block; "Keywords:" goes to the PDF's metadata only.
TITLE_LABELS = (("author", "Author:"), ("affiliation", "Affiliation:"), ("email", "Email:"),
                ("code", "Code:"), ("keywords", "Keywords:"))


def front_matter(text: str) -> dict:
    """The title block of the draft: title, the labelled lines, and placeholders left."""
    lines = text.splitlines()
    out: dict = {"title": None, "author": None, "affiliation": None, "email": None,
                 "code": None, "keywords": None}
    for line in lines:
        if line.startswith("## "):
            break
        if line.startswith("# ") and out["title"] is None:
            out["title"] = line[2:].strip()
        for key, label in TITLE_LABELS:
            if line.startswith(label):
                out[key] = line[len(label):].strip()
    out["placeholders"] = [f"{k}: {v}" for k, _ in TITLE_LABELS
                           if (v := out[k]) and any(p in v for p in PLACEHOLDERS)]
    out["missing"] = [k for k in ("title", "author", "code") if not out[k]]
    return out


TODO_RE = re.compile(r"TODO\([^)]*\)")


def todo_markers(text: str) -> list[dict]:
    """The draft's TODO(...) markers after its title block, which the PDF would print."""
    out = []
    in_body = False
    for k, line in enumerate(text.splitlines(), 1):
        in_body = in_body or line.startswith("## ")
        if in_body:
            for m in TODO_RE.finditer(line):
                out.append({"line": k, "marker": m.group(0), "context": line.strip()[:100]})
    return out


# --------------------------------------------------------------------- figures

IMAGE_RE = re.compile(r"!\[[^\]]*\]\(fig/([A-Za-z0-9_\-]+)\.png\)")


def figure_names(text: str) -> list[str]:
    return IMAGE_RE.findall(text)


def svg_width_in(svg: Path) -> float | None:
    head = svg.read_text(encoding="utf-8", errors="replace")[:4000]
    m = re.search(r"<svg\b[^>]*\bwidth=\"([\d.]+)(pt|px|in)?\"", head)
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2) or "px"
    return {"pt": v / 72.0, "in": v, "px": v / 96.0}[unit]


def png_size(png: Path) -> tuple[int, int, float | None]:
    """Pixel width, height and dpi (from pHYs) of a PNG."""
    b = png.read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise BuildError(f"{png} is not a PNG")
    w, h = struct.unpack(">II", b[16:24])
    dpi = None
    i = b.find(b"pHYs")
    if i > 0:
        x, _y, unit = struct.unpack(">IIB", b[i + 4:i + 13])
        if unit == 1:
            dpi = x * 0.0254
    return w, h, dpi


def prepare_figures(names: list[str], src: Path, dest: Path, mode: str) -> dict:
    """Copy or convert each figure into dest; returns name -> {file, width, kind}."""
    dest.mkdir(parents=True, exist_ok=True)
    rsvg = find_tool("rsvg-convert", "RSVG_CONVERT") if mode != "png" else None
    cairosvg = None
    if mode != "png" and not rsvg:
        try:
            import cairosvg  # type: ignore  # noqa: F401
        except ImportError:
            cairosvg = None
    if mode == "vector" and not (rsvg or cairosvg):
        raise BuildError("--figures vector needs rsvg-convert or the cairosvg module")
    out = {}
    for name in dict.fromkeys(names):
        png, svg = src / f"{name}.png", src / f"{name}.svg"
        if not png.exists():
            raise BuildError(f"figure {rel(png)} is missing")
        w_px, _h, dpi = png_size(png)
        width = svg_width_in(svg) if svg.exists() else None
        if width is None:
            width = w_px / (dpi or 200.0)
        if (rsvg or cairosvg) and svg.exists():
            target = dest / f"{name}.pdf"
            if rsvg:
                subprocess.run([rsvg, "-f", "pdf", "-o", str(target), str(svg)], check=True)
                kind = "pdf (rsvg-convert)"
            else:
                cairosvg.svg2pdf(url=str(svg), write_to=str(target))
                kind = "pdf (cairosvg)"
        else:
            target = dest / f"{name}.png"
            shutil.copyfile(png, target)
            kind = f"png ({dpi or 'unknown'} dpi)" if dpi is None else f"png ({dpi:.0f} dpi)"
        out[name] = {"file": f"fig/{target.name}", "width": f"{width:.3f}in", "kind": kind,
                     "source": rel(svg if target.suffix == ".pdf" else png)}
    return out


def figures_md_placement(text: str) -> dict:
    """figures.md's placement table: draft label -> (section, drawn figure, file stem)."""
    out = {}
    for m in re.finditer(r"^\| Figure ([A-Z]?\d+) \| ([^|]+) \| (D\d+) \| `([^`]+)` \|", text, re.M):
        out[m.group(1)] = (m.group(2).strip(), m.group(3), m.group(4))
    return out


def figures_md_captions(text: str) -> dict:
    """figures.md's full captions by drawn figure: D1 -> caption Markdown."""
    out = {}
    for m in re.finditer(r"^### (D\d+)\. .*?\n\n\*\*Caption\.\*\* (.+?)\n", text, re.M | re.S):
        out[m.group(1)] = m.group(2).strip()
    return out


def check_placement(draft: str, figures_md: str) -> list[str]:
    """Draft figure labels and files against figures.md's placement table."""
    table = figures_md_placement(figures_md)
    found = {}
    section = None
    lines = draft.splitlines()
    for k, line in enumerate(lines):
        m = re.match(r"^#{2,3} (?:Appendix )?([A-Z]?\d*(?:\.\d+)?)[ :]", line)
        if m and m.group(1):
            section = m.group(1)
        mi = IMAGE_RE.search(line)
        if mi:
            nxt = next((x for x in lines[k + 1:] if x.strip()), "")
            ml = re.match(r"^\*Figure ([A-Z]?\d+)\.\*", nxt)
            if ml:
                found[ml.group(1)] = (section, mi.group(1))
    problems = []
    for label, (sec, _d, stem) in table.items():
        if label not in found:
            problems.append(f"Figure {label} ({stem}) is in figures.md but not in the draft")
            continue
        dsec, dstem = found[label]
        if dstem != stem:
            problems.append(f"Figure {label}: the draft shows {dstem}, figures.md says {stem}")
        want = sec.replace("§", "").replace("App ", "")
        if dsec != want:
            problems.append(f"Figure {label}: in {dsec} in the draft, figures.md says {sec}")
    for label in found:
        if label not in table:
            problems.append(f"Figure {label} is in the draft but not in figures.md's table")
    return problems


# ----------------------------------------------------------------------- the log

OVERFULL_RE = re.compile(
    r"^Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\) (?:in paragraph|in alignment|"
    r"detected|has occurred while \\output is active)(?: at lines (\d+)--(\d+))?(.*)$", re.M)


def parse_log(log: str) -> dict:
    """Overfull boxes, missing glyphs, undefined references and other warnings."""
    lines = log.splitlines()
    overfull = []
    for k, line in enumerate(lines):
        m = OVERFULL_RE.match(line)
        if m:
            ctx = lines[k + 1].strip() if k + 1 < len(lines) else ""
            overfull.append({
                "pt": float(m.group(1)),
                "lines": f"{m.group(2)}--{m.group(3)}" if m.group(2) else None,
                "where": "alignment" if "alignment" in line else "paragraph",
                "context": ctx[:160],
            })
    underfull = len(re.findall(r"^Underfull \\[hv]box", log, re.M))
    missing = sorted(set(re.findall(r"Missing character: There is no (.+?) in font", log)))
    undefined = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
    hyper = len(re.findall(r"Token not allowed in a PDF string", log))
    other = []
    for m in re.finditer(r"^(LaTeX (?:Font )?Warning|Package \w+ Warning): (.*)$", log, re.M):
        msg = m.group(2).strip()
        if "Token not allowed" in msg or "undefined" in msg and "Reference" in msg:
            continue
        other.append(f"{m.group(1)}: {msg}"[:200])
    pages = None
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    if m:
        pages = int(m.group(1))
    rerun = bool(re.search(r"Rerun to get|Label\(s\) may have changed|has changed\. Rerun|"
                           r"There were undefined references", log))
    return {"overfull": overfull, "underfull": underfull, "missing_glyphs": missing,
            "undefined_refs": undefined, "pdf_string_tokens": hyper,
            "other_warnings": sorted(set(other)), "pages": pages, "rerun": rerun}


AUX_LABEL_RE = re.compile(r"\\newlabel\{([^}]+)\}\{\{((?:[^{}]|\{[^{}]*\})*)\}\{([^{}]*)\}")


def aux_pages(aux: str) -> dict:
    """Label -> page number from a LaTeX .aux file (hyperref's five-field form too)."""
    out = {}
    for m in AUX_LABEL_RE.finditer(aux):
        page = m.group(3).strip()
        if page.isdigit():
            out[m.group(1)] = int(page)
    return out


def page_summary(labels: dict, pages: int | None) -> dict:
    total = pages or labels.get("report:end")
    app = labels.get("report:appendix-start")
    refs = labels.get("references")
    out = {"total": total}
    if app and total:
        out["main_text"] = [1, app - 1]
        out["appendices"] = [app, total]
        out["main_text_pages"] = app - 1
        out["appendix_pages"] = total - app + 1
    if refs and app:
        out["references"] = [refs, app - 1]
    return out


# ----------------------------------------------------------------- text fidelity

NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")
WORD_RE = re.compile(r"[^\W\d_]+")
GREEK_QUESTION_MARK = "\u037e"
MINUS_SIGN = "\u2212"
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def pandoc_reader() -> str:
    """The input format of defaults.yaml, so the check reads the draft as the build does."""
    m = re.search(r"^from:\s*(\S+)", (KIT / "defaults.yaml").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "gfm"


# What TeX math holds that prints nothing, dropped before the commands are: a comment
# (an unescaped % to the end of the line); the column spec and position of an array or
# an alignment (\begin{array}{cc}, \begin{alignat}{2}, \begin{aligned}[t]); a row's
# extra space (\\[2pt]); a bare dimension (\kern2pt, \mskip 3mu); and the commands
# whose first braced argument prints nothing (spacing, labels and their references,
# environment names, colours), with that argument.
MATH_COMMENT = re.compile(r"(?<!\\)%[^\n]*")
MATH_ENV_ARGS = re.compile(
    r"\\begin\s*\{(?:(?:array|subarray|alignat\*?|alignedat|tabular)\}"
    r"(?:\s*\[[^\]]*\])?\s*\{[^{}]*\}|(?:aligned|gathered)\}\s*\[[^\]]*\])")
MATH_ROW_SPACE = re.compile(r"\\\\\s*\[[^\]]*\]")
MATH_DIMEN = re.compile(r"\\(?:kern|mkern|mskip|hskip|vskip|hspace\*?|vspace\*?|mspace)"
                        r"\s*-?\s*(?:\d+(?:\.\d*)?|\.\d+)\s*[a-z]{2}")
MATH_SILENT_ARG = re.compile(
    r"\\(?:begin|end|label|ref|eqref|pageref|hspace\*?|vspace\*?|kern|mkern|mskip|mspace|"
    r"hskip|phantom|hphantom|vphantom|color|textcolor|operatornamewithlimits)\s*\{[^{}]*\}")
MATH_ESCAPED = {"%": "%", "$": "$", "_": "_", "&": "&", "#": "#", "{": "{", "}": "}"}


def math_text(tex: str) -> str:
    """A TeX math string as its printed words and numbers, for the text check.

    Command names (\\mu, \\mathrm, \\operatorname, \\times) are dropped, so they are never
    expected as words: Greek letters and symbols print as glyphs the check does not
    look for. The arguments of \\text{...}, \\mathrm{...}, \\operatorname{...} and the like
    print, so their words are kept. Comments, column specs, spacing, labels and
    environment names print nothing and are dropped (see MATH_SILENT_ARG). Braces, ^,
    _, & and \\\\ become spaces, so B_{31} gives "31" and \\tfrac{1}{2} gives "1" and
    "2", never "12". \\% is "%", and a "-" stays a hyphen (the check drops hyphens on
    both sides)."""
    s = MATH_COMMENT.sub(" ", tex)
    s = MATH_ENV_ARGS.sub(" ", s)
    s = MATH_ROW_SPACE.sub(" ", s)
    s = MATH_DIMEN.sub(" ", s)
    s = MATH_SILENT_ARG.sub(" ", s)
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s)
    s = re.sub(r"\\(.)", lambda m: MATH_ESCAPED.get(m.group(1), " "), s)
    s = re.sub(r"[{}^_&~]", " ", s)
    return " " + re.sub(r"\s+", " ", s).strip() + " "


def _plain_math(node):
    """A pandoc JSON tree with every Math element replaced by its math_text."""
    if isinstance(node, list):
        return [_plain_math(x) for x in node]
    if isinstance(node, dict):
        if node.get("t") == "Math":
            return {"t": "Str", "c": math_text(node["c"][1])}
        return {k: _plain_math(v) for k, v in node.items()}
    return node


def draft_plain(pandoc: str, text: str) -> str:
    """The draft as pandoc's plain text, less what the PDF does not print by design:
    image alt text, the "Author:", "Affiliation:" and "Email:" labels of the title block, and its
    "Keywords:" line (metadata only). TeX math is read as math_text gives it, not as
    pandoc's plain writer renders it."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"^(?:Author|Affiliation|Email):", "", text, flags=re.M)
    text = re.sub(r"^Keywords:.*$", "", text, flags=re.M)
    p = subprocess.run([pandoc, "-f", pandoc_reader(), "-t", "json"],
                       input=text, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise BuildError(f"pandoc (json) failed: {p.stderr.strip()}")
    doc = _plain_math(json.loads(p.stdout))
    p = subprocess.run([pandoc, "-f", "json", "-t", "plain", "--wrap=none"],
                       input=json.dumps(doc), capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise BuildError(f"pandoc (plain text) failed: {p.stderr.strip()}")
    return p.stdout


def pdf_plain(pdftotext: str, pdf: Path) -> str:
    """The PDF's text without the page numbers in the footers."""
    p = subprocess.run([pdftotext, "-enc", "UTF-8", str(pdf), "-"], capture_output=True,
                       text=True, timeout=120)
    if p.returncode != 0:
        raise BuildError(f"pdftotext failed: {p.stderr.strip()}")
    pages = []
    for page in p.stdout.split("\f"):
        lines = page.rstrip().splitlines()
        if lines and lines[-1].strip().isdigit():
            lines = lines[:-1]
        pages.append("\n".join(lines))
    return "\n".join(pages)


def _normalise(s: str) -> str:
    """Both sides of the text check: NFKC (ligatures, the mathematical italic letters
    and sub- and superscript digits of typeset math), the minus sign read as a hyphen,
    soft hyphens and hyphens dropped, a word broken at a line end joined."""
    s = unicodedata.normalize("NFKC", s)
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace(MINUS_SIGN, "-")
    s = s.replace("­", "")
    s = re.sub(r"[-‐‑]\n\s*", "", s)   # a hyphen at a line end, TeX's or the text's
    return s.replace("-", "").replace("‐", "").replace("‑", "")


def text_check(draft_txt: str, pdf_txt: str) -> dict:
    """Every number and word of the draft against the PDF's text, as multisets.

    Hyphens are dropped on both sides (TeX hyphenates at line ends; the draft's
    "test-like" may break there). An occurrence the PDF's text lacks, as pdftotext splits
    it, counts as split, not missing, as far as the token occurs more often in that text
    with all whitespace removed than as a token: inline code and URLs break inside a
    token (a hash across two lines). Short tokens (a digit, "a") are therefore weakly
    checked; long numbers and words are not, and a repeated number the PDF drops once
    is caught.

    "greek_question_marks" counts the U+037E the PDF's text has beyond the draft's: a
    font whose ToUnicode map sends ";" there makes every copied or searched semicolon
    the Greek question mark (template.tex turns on XeTeX's ActualText against it).
    """
    d, p = _normalise(draft_txt), _normalise(pdf_txt)
    compact = re.sub(r"\s+", "", p)
    out = {}
    for kind, rx, fold in (("numbers", NUMBER_RE, False), ("words", WORD_RE, True)):
        def tokens(s):
            return Counter(t.lower() if fold else t for t in rx.findall(s))
        want, have = tokens(d), tokens(p)
        short = want - have
        hay = compact.lower() if fold else compact
        missing = {}
        for t, n in sorted(short.items()):
            split = max(0, hay.count(t) - have[t])
            if n > split:
                missing[t] = n - split
        out[kind] = sum(want.values())
        out[f"{kind}_split"] = sum(short.values()) - sum(missing.values())
        out[f"{kind}_missing"] = missing
    out["greek_question_marks"] = max(0, pdf_txt.count(GREEK_QUESTION_MARK)
                                      - draft_txt.count(GREEK_QUESTION_MARK))
    return out


# --------------------------------------------------------------------- the build

def run_pandoc(pandoc: str, draft: Path, tex: Path, config: Path) -> str:
    env = dict(os.environ, PAIEC_REPORT_CONFIG=str(config))
    cmd = [pandoc, str(draft), "--defaults", str(KIT / "defaults.yaml"), "-o", str(tex)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tex.parent))
    if p.returncode != 0:
        raise BuildError(f"pandoc failed ({p.returncode}):\n{p.stderr.strip()}")
    return p.stderr


def run_xelatex(xelatex: str, tex: Path, epoch: int) -> tuple[str, int]:
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch), FORCE_SOURCE_DATE="1",
               max_print_line="10000", error_line="254", half_error_line="238")
    cmd = [xelatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
           tex.name]
    log_path = tex.with_suffix(".log")
    for run in range(1, MAX_LATEX_RUNS + 1):
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tex.parent),
                           errors="replace")
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if p.returncode != 0:
            errs = [ln for ln in log.splitlines() if ln.startswith("!") or ".tex:" in ln]
            raise BuildError("xelatex failed:\n" + "\n".join(errs[-12:] or log.splitlines()[-30:]))
        if run >= 2 and not parse_log(log)["rerun"]:
            return log, run
    return log, MAX_LATEX_RUNS


def build(draft: Path = DRAFT, out_dir: Path = OUT, *, figures: str = "auto",
          captions: str = "draft", epoch: int | None = None, tex_only: bool = False,
          quiet: bool = False) -> dict:
    """Build the PDF; returns the build report (also written to out_dir/build.json)."""
    pandoc = find_tool("pandoc", "PANDOC")
    if not pandoc:
        raise BuildError("pandoc not found (set PANDOC)")
    xelatex = find_tool("xelatex", "XELATEX")
    if not xelatex and not tex_only:
        raise BuildError("xelatex not found (set XELATEX)")
    if xelatex and not tex_only and (lacking := missing_latex_packages(xelatex)):
        raise BuildError("the TeX installation lacks these packages of template.tex: "
                         + ", ".join(lacking))
    draft = Path(draft).resolve()
    out_dir = Path(out_dir).resolve()
    build_dir = out_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    text = draft.read_text(encoding="utf-8")
    fm = front_matter(text)
    if fm["missing"]:
        raise BuildError(f"the draft's title block lacks: {', '.join(fm['missing'])}")
    names = figure_names(text)
    figs = prepare_figures(names, draft.parent / "fig", build_dir / "fig", figures)
    fmd = FIGURES_MD.read_text(encoding="utf-8") if FIGURES_MD.exists() else ""
    placement = check_placement(text, fmd) if fmd and draft == DRAFT.resolve() else []
    caption_map = {}
    if captions == "full":
        full = figures_md_captions(fmd)
        for label, (_sec, d, _stem) in figures_md_placement(fmd).items():
            if d in full:
                caption_map[label] = full[d]
    config = build_dir / "filter_config.json"
    filter_log = build_dir / "filter_report.json"
    config.write_text(json.dumps({
        "figures": {n: {"file": f["file"], "width": f["width"]} for n, f in figs.items()},
        "captions": caption_map, "log": str(filter_log)}, indent=1), encoding="utf-8")
    tex = build_dir / "report.tex"
    pandoc_stderr = run_pandoc(pandoc, draft, tex, config)
    flt = json.loads(filter_log.read_text(encoding="utf-8"))
    epoch = source_date_epoch() if epoch is None else epoch
    report = {
        "draft": rel(draft),
        "inputs": {rel(p): sha256(p) for p in
                   [draft, KIT / "defaults.yaml", KIT / "template.tex", KIT / "report.lua",
                    Path(__file__).resolve()]
                   + sorted({ROOT / f["source"] for f in figs.values()})},
        "tools": {"pandoc": tool_version(pandoc), "xelatex": tool_version(xelatex) if xelatex else None},
        "git": {"head": git("rev-parse", "--short", "HEAD"),
                "inputs_dirty": bool(git("status", "--porcelain", "--", "docs/report",
                                         "tools/report_pdf", "tools/build_report_pdf.py"))},
        "source_date_epoch": epoch,
        "front_matter": fm,
        "todo_markers": todo_markers(text),
        "figures": figs,
        "figure_placement": placement,
        "captions": captions,
        # pandoc's JSON writes an empty Lua list as {}
        "filter": {k: (flt.get(k) or ({} if k == "links" else [])) for k in
                   ("links", "unresolved", "warnings", "tables", "figures", "references")},
        "pandoc_stderr": [ln for ln in pandoc_stderr.splitlines() if ln.strip()],
    }
    if tex_only:
        report["tex"] = str(tex)
        (out_dir / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return report
    log, runs = run_xelatex(xelatex, tex, epoch)
    parsed = parse_log(log)
    labels = aux_pages(tex.with_suffix(".aux").read_text(encoding="utf-8", errors="replace"))
    pdf_built = tex.with_suffix(".pdf")
    pdf = out_dir / PDF_NAME
    shutil.copyfile(pdf_built, pdf)
    size = pdf.stat().st_size
    pdftotext = find_tool("pdftotext", "PDFTOTEXT")
    if pdftotext:
        checked = text_check(draft_plain(pandoc, text), pdf_plain(pdftotext, pdf))
    else:
        checked = {"skipped": "pdftotext not found (set PDFTOTEXT)"}
    report.update({
        "pdf": str(pdf),
        "pdf_sha256": sha256(pdf),
        "pdf_bytes": size,
        "pdf_within_limit": size <= MAX_PDF_BYTES,
        "latex_runs": runs,
        "pages": page_summary(labels, parsed["pages"]),
        "label_pages": {k: v for k, v in labels.items()
                        if k.startswith(("report:", "fig-", "tab-", "references"))
                        or k in ("sec-1", "sec-7")},
        "log": {k: v for k, v in parsed.items() if k != "rerun"},
        "text_check": checked,
    })
    (out_dir / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if not quiet:
        print(summary(report))
    return report


def problems(report: dict, overfull_limit: float = OVERFULL_LIMIT_PT) -> list[str]:
    """What --strict fails on."""
    out = []
    log = report.get("log", {})
    big = [o for o in log.get("overfull", []) if o["pt"] > overfull_limit]
    if big:
        out.append(f"{len(big)} overfull boxes above {overfull_limit}pt")
    if log.get("missing_glyphs"):
        out.append("missing glyphs: " + ", ".join(log["missing_glyphs"]))
    if log.get("undefined_refs"):
        out.append("undefined references: " + ", ".join(log["undefined_refs"]))
    if report.get("filter", {}).get("unresolved"):
        out.append(f"{len(report['filter']['unresolved'])} unresolved cross-references")
    if report.get("front_matter", {}).get("placeholders"):
        out.append("title block placeholders: " + "; ".join(report["front_matter"]["placeholders"]))
    if report.get("todo_markers"):
        out.append(f"{len(report['todo_markers'])} TODO markers in the draft's body (lines "
                   + ", ".join(str(t["line"]) for t in report["todo_markers"]) + ")")
    if report.get("pdf_bytes") and not report.get("pdf_within_limit"):
        out.append(f"PDF is {report['pdf_bytes']} bytes, over {MAX_PDF_BYTES}")
    if report.get("figure_placement"):
        out.append("figure placement differs from figures.md")
    tc = report.get("text_check") or {}
    for kind in ("numbers", "words"):
        if tc.get(f"{kind}_missing"):
            out.append(f"{kind} of the draft missing from the PDF's text: "
                       + ", ".join(sorted(tc[f"{kind}_missing"]))[:300])
    if tc.get("greek_question_marks"):
        out.append(f"{tc['greek_question_marks']} semicolons in the PDF's text read as U+037E")
    return out


def summary(report: dict) -> str:
    pg = report.get("pages", {})
    lines = []
    if "pdf" in report:
        lines.append(f"PDF: {rel(Path(report['pdf']))} ({report['pdf_bytes'] / 1e6:.2f} MB, {pg.get('total')} pages, "
                     f"{report['latex_runs']} XeLaTeX runs)")
        if "main_text" in pg:
            refs = pg.get("references")
            lines.append(f"  main text: pages {pg['main_text'][0]}-{pg['main_text'][1]}"
                         + (f" (references {refs[0]}-{refs[1]})" if refs else "")
                         + f"; appendices: pages {pg['appendices'][0]}-{pg['appendices'][1]}")
    kinds = sorted({f["kind"] for f in report["figures"].values()})
    lines.append(f"  figures: {len(report['figures'])} ({', '.join(kinds)})")
    t = report["filter"].get("tables") or []
    sizes = {}
    for x in t:
        sizes[x["size"]] = sizes.get(x["size"], 0) + 1
    kept = sum(1 for x in t if x.get("kept_together"))
    lines.append(f"  tables: {len(t)} ({', '.join(f'{v} {k}' for k, v in sorted(sizes.items()))}; "
                 f"{kept} kept on one page, {len(t) - kept} longtables that may break)")
    lk = report["filter"].get("links") or {}
    lines.append("  links: " + ", ".join(f"{v} {k}" for k, v in lk.items()))
    log = report.get("log", {})
    if log:
        ov = log["overfull"]
        lines.append(f"  overfull boxes: {len(ov)}"
                     + (f" (largest {max(o['pt'] for o in ov):.2f}pt)" if ov else "")
                     + f"; underfull: {log['underfull']}")
        for o in sorted(ov, key=lambda o: -o["pt"])[:10]:
            lines.append(f"    {o['pt']:.2f}pt {o['where']} at report.tex lines {o['lines']}: "
                         f"{o['context'][:90]}")
        for key, label in (("missing_glyphs", "missing glyphs"),
                           ("undefined_refs", "undefined references")):
            if log.get(key):
                lines.append(f"  {label}: {', '.join(log[key])}")
        if log.get("other_warnings"):
            lines.append(f"  other LaTeX warnings: {len(log['other_warnings'])}")
            for w in log["other_warnings"][:8]:
                lines.append(f"    {w}")
    tc = report.get("text_check") or {}
    if "numbers" in tc:
        miss = {k: len(tc[f"{k}_missing"]) for k in ("numbers", "words")}
        lines.append(f"  text: {tc['numbers']} numbers and {tc['words']} words of the draft; "
                     f"missing from the PDF: {miss['numbers']} numbers, {miss['words']} words "
                     f"({tc['numbers_split'] + tc['words_split']} split across lines)")
        for kind in ("numbers", "words"):
            if tc[f"{kind}_missing"]:
                lines.append(f"    missing {kind}: "
                             + ", ".join(f"{t} x{n}" for t, n in tc[f"{kind}_missing"].items())[:300])
        if tc.get("greek_question_marks"):
            lines.append(f"    U+037E (Greek question mark) for ';': {tc['greek_question_marks']}")
    elif tc:
        lines.append(f"  text check skipped: {tc.get('skipped')}")
    for w in report["filter"].get("warnings") or []:
        lines.append(f"  filter: {w}")
    un = report["filter"].get("unresolved") or []
    if un:
        lines.append(f"  unresolved references (left as text): "
                     + ", ".join(sorted({u['text'] for u in un})))
    for p in report.get("figure_placement") or []:
        lines.append(f"  figures.md: {p}")
    for p in report["front_matter"].get("placeholders") or []:
        lines.append(f"  placeholder in the title block: {p}")
    todos = report.get("todo_markers") or []
    if todos:
        lines.append(f"  TODO markers in the draft's body, printed in the PDF: {len(todos)} (draft lines "
                     + ", ".join(str(t["line"]) for t in todos) + ")")
    return "\n".join(lines)


# ----------------------------------------------------------------- page renders

def render_pages(report: dict, which: str, dest: Path, dpi: int = 80) -> list[Path]:
    """PNGs of chosen pages (comma list, or 'auto': title, Table 1, Figure 5, appendix)."""
    pdftoppm = find_tool("pdftoppm", "PDFTOPPM")
    if not pdftoppm:
        raise BuildError("pdftoppm not found (set PDFTOPPM)")
    lp = report.get("label_pages", {})
    if which == "auto":
        pages = [1, lp.get("tab-1"), lp.get("fig-5"), lp.get("report:appendix-start")]
    else:
        pages = [int(x) for x in which.split(",") if x.strip()]
    pages = [p for p in dict.fromkeys(pages) if p]
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("page-*.png"):   # renders of an earlier build
        old.unlink()
    out = []
    for p in pages:
        stem = dest / f"page-{p:03d}"
        subprocess.run([pdftoppm, "-r", str(dpi), "-png", "-singlefile", "-f", str(p), "-l", str(p),
                        report["pdf"], str(stem)], check=True)
        out.append(stem.with_suffix(".png"))
    return out


# ------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", type=Path, default=DRAFT, help="the Markdown source")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (dist/report)")
    ap.add_argument("--figures", choices=("auto", "vector", "png"), default="auto",
                    help="auto: vector PDF from the SVGs if rsvg-convert or cairosvg is "
                         "available, else the 200-dpi PNGs")
    ap.add_argument("--captions", choices=("draft", "full"), default="draft",
                    help="draft: the draft's own captions; full: figures.md's long ones")
    ap.add_argument("--tex-only", action="store_true", help="stop after pandoc")
    ap.add_argument("--render", metavar="PAGES",
                    help="also write PNGs of these pages ('auto' or e.g. 1,12,40) to OUT/pages")
    ap.add_argument("--check-deterministic", action="store_true",
                    help="build twice in temporary directories and compare the PDFs")
    ap.add_argument("--strict", action="store_true",
                    help=f"exit 2 on overfull boxes above {OVERFULL_LIMIT_PT}pt, missing glyphs, "
                         "undefined or unresolved references, title-block placeholders, "
                         "TODO(...) markers in the draft's body, "
                         "a figure placement that differs from figures.md, a number or word "
                         "of the draft missing from the PDF's text, a semicolon read as "
                         "U+037E, or a PDF over 50 MB")
    args = ap.parse_args(argv)
    try:
        report = build(args.draft, args.out, figures=args.figures, captions=args.captions,
                       tex_only=args.tex_only)
        if args.tex_only:
            print(f"LaTeX: {report['tex']}")
            return 0
        if args.render:
            for p in render_pages(report, args.render, Path(args.out) / "pages"):
                print(f"  rendered {p}")
        if args.check_deterministic:
            with tempfile.TemporaryDirectory() as tmp:
                a = build(args.draft, Path(tmp) / "a", figures=args.figures,
                          captions=args.captions, epoch=report["source_date_epoch"], quiet=True)
                b = build(args.draft, Path(tmp) / "b", figures=args.figures,
                          captions=args.captions, epoch=report["source_date_epoch"], quiet=True)
                same = a["pdf_sha256"] == b["pdf_sha256"] == report["pdf_sha256"]
                print(f"  deterministic: {'yes' if same else 'NO'} (sha256 {report['pdf_sha256'][:16]}…)")
                if not same:
                    return 3
    except BuildError as e:
        print(f"build_report_pdf: {e}", file=sys.stderr)
        return 1
    if args.strict:
        bad = problems(report)
        if bad:
            for b in bad:
                print(f"strict: {b}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
