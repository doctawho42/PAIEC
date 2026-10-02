"""tools/build_report_pdf.py: docs/report/draft.md builds into the report PDF.

The pure-Python parts (title block, LaTeX log and .aux parsing, the text check, figure
placement against docs/report/figures.md) run everywhere. The Lua filter tests need
pandoc, and the PDF builds need pandoc and XeLaTeX with the template's packages; they
are skipped when those are missing (looked up as the build does: PATH, then
/opt/homebrew/bin, /opt/local/bin, /usr/local/bin and /Library/TeX/texbin; PANDOC and
XELATEX override). No data download and no network are needed.
"""
import hashlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "docs" / "report" / "draft.md"
CITATION = ROOT / "CITATION.cff"
FIGURES_MD = ROOT / "docs" / "report" / "figures.md"
FIG = ROOT / "docs" / "report" / "fig"


def _load():
    spec = importlib.util.spec_from_file_location("build_report_pdf",
                                                  ROOT / "tools" / "build_report_pdf.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_report_pdf"] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load()
PANDOC = B.find_tool("pandoc", "PANDOC")
XELATEX = B.find_tool("xelatex", "XELATEX")
LACKING = B.missing_latex_packages(XELATEX) if XELATEX else None

needs_pandoc = pytest.mark.skipif(not PANDOC, reason="pandoc not found (set PANDOC)")
needs_latex = pytest.mark.skipif(
    not (PANDOC and XELATEX and not LACKING),
    reason="pandoc not found (set PANDOC)" if not PANDOC
    else "xelatex not found (set XELATEX)" if not XELATEX
    else f"the TeX installation lacks {', '.join(LACKING or [])}")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _title_block(text: str) -> str:
    return text.split("\n## ", 1)[0]


def _cff_keywords(text: str) -> list[str]:
    """The top-level keywords list of CITATION.cff (no YAML parser needed)."""
    block = text.split("\nkeywords:\n", 1)[1]
    out = []
    for line in block.splitlines():
        if not line.startswith("  - "):
            break
        out.append(line[4:].strip())
    return out


# ------------------------------------------------------------------ pure Python

def test_title_block_of_the_draft():
    text = DRAFT.read_text(encoding="utf-8")
    fm = B.front_matter(text)
    assert fm["missing"] == []
    assert fm["author"] == "Nikita L. Polomoshnov"
    assert fm["code"].startswith("https://github.com/")
    assert fm["affiliation"] == "Moscow State University"
    assert fm["email"] == "nikitapol@fbb.msu.ru"
    assert EMAIL_RE.findall(_title_block(text)) == ["nikitapol@fbb.msu.ru"], "only the team's e-mail"


def test_keywords_have_one_source():
    """The draft's "Keywords:" line fills the PDF metadata; CITATION.cff repeats it."""
    kw = B.front_matter(DRAFT.read_text(encoding="utf-8"))["keywords"]
    assert kw and "TODO" not in kw
    assert [k.strip() for k in kw.split(",")] == _cff_keywords(CITATION.read_text(encoding="utf-8"))


def test_front_matter_reports_placeholders_and_gaps():
    # the repository placeholder in two pieces, so that a search for it finds only the
    # places to fill (docs/release.md, section 8)
    text = ("# T\n\nsub\n\nAuthor: A. B. Cee\n\nAffiliation: TODO(team): affiliation\n\n"
            "Code: https://github.com/" + "OWNER" + "/REPO\n\nKeywords: a, b\n\n"
            "## Abstract\n\nx\n")
    fm = B.front_matter(text)
    assert fm["title"] == "T" and fm["author"] == "A. B. Cee" and fm["keywords"] == "a, b"
    assert len(fm["placeholders"]) == 2
    assert len(B.front_matter(text.replace("a, b", "TODO(team)"))["placeholders"]) == 3
    assert B.front_matter("# T\n\n## Abstract\n")["missing"] == ["author", "code"]


def test_figures_match_figures_md_and_exist():
    draft = DRAFT.read_text(encoding="utf-8")
    fmd = FIGURES_MD.read_text(encoding="utf-8")
    assert B.check_placement(draft, fmd) == []
    names = B.figure_names(draft)
    assert len(names) == len(set(names)) == len(B.figures_md_placement(fmd))
    for n in names:
        assert (FIG / f"{n}.png").exists() and (FIG / f"{n}.svg").exists(), n
        w, h, dpi = B.png_size(FIG / f"{n}.png")
        assert w > 0 and h > 0 and dpi and round(dpi) == 200, n
        assert 1.0 < B.svg_width_in(FIG / f"{n}.svg") <= 6.5, n   # within the 6.5in text width


def test_parse_log():
    log = "\n".join([
        "Overfull \\hbox (4.5pt too wide) in paragraph at lines 10--12",
        "[]\\TU/X/m/n/10 some text",
        "Overfull \\hbox (0.8pt too wide) in alignment at lines 20--30",
        "[] []",
        "Underfull \\hbox (badness 10000) in paragraph at lines 3--4",
        "Missing character: There is no Ж in font TeX Gyre Termes!",
        "LaTeX Warning: Reference `sec-9' on page 3 undefined on input line 7.",
        "LaTeX Warning: Float too large for page by 4.0pt on input line 9.",
        "Output written on report.pdf (12 pages).",
    ])
    p = B.parse_log(log)
    assert [o["pt"] for o in p["overfull"]] == [4.5, 0.8]
    assert p["overfull"][1]["where"] == "alignment" and p["overfull"][0]["lines"] == "10--12"
    assert p["underfull"] == 1 and p["missing_glyphs"] == ["Ж"]
    assert p["undefined_refs"] == ["sec-9"] and p["pages"] == 12
    assert p["other_warnings"] == ["LaTeX Warning: Float too large for page by 4.0pt on input line 9."]
    assert p["rerun"] is False
    assert B.parse_log("LaTeX Warning: Label(s) may have changed. Rerun to get it right.")["rerun"]


def test_aux_pages_and_page_summary():
    aux = ("\\newlabel{sec-1}{{}{1}{}{section*.1}{}}\n"
           "\\newlabel{references}{{}{28}{}{section*.40}{}}\n"
           "\\newlabel{report:appendix-start}{{}{31}{}{section*.41}{}}\n"
           "\\newlabel{report:end}{{}{74}{}{}{}}\n")
    labels = B.aux_pages(aux)
    assert labels == {"sec-1": 1, "references": 28, "report:appendix-start": 31, "report:end": 74}
    s = B.page_summary(labels, 74)
    assert s["main_text"] == [1, 30] and s["appendices"] == [31, 74]
    assert s["references"] == [28, 30] and s["appendix_pages"] == 44


def test_text_check_finds_what_the_pdf_lost():
    draft = "The test-like gain is 0.042 ALC (sha 0de18448abcdef1234567); 0.042 again."
    pdf = "The test-\nlike gain is 0.042 ALC (sha 0de18448abcd\nef1234567);"
    tc = B.text_check(draft, pdf)
    assert tc["numbers_missing"] == {"0.042": 1}   # the second 0.042 is gone
    assert tc["words_missing"] == {"again": 1}
    ok = B.text_check(draft, pdf + " 0.042 again.")
    assert ok["numbers_missing"] == {} and ok["words_missing"] == {}
    assert tc["greek_question_marks"] == 0


def test_text_check_flags_semicolons_read_as_greek_question_marks():
    tc = B.text_check("one; two; three", "one\u037e two\u037e three")
    assert tc["greek_question_marks"] == 2 and tc["words_missing"] == {}
    assert any("U+037E" in p for p in B.problems({"text_check": tc}))
    assert B.text_check("ερώτηση\u037e", "ερώτηση\u037e")["greek_question_marks"] == 0


def test_todo_markers_after_the_title_block():
    text = "# T\n\nAffiliation: TODO(team): x\n\n## 1 A\n\nDone. `TODO(record)` and TODO(team) here.\n"
    assert [(t["line"], t["marker"]) for t in B.todo_markers(text)] == [
        (7, "TODO(record)"), (7, "TODO(team)")]


def test_source_date_epoch():
    assert B.source_date_epoch({"SOURCE_DATE_EPOCH": "1700000000"}) == 1700000000
    with pytest.raises(B.BuildError):
        B.source_date_epoch({"SOURCE_DATE_EPOCH": "yesterday"})


def test_template_packages_are_listed():
    pkgs = B.latex_packages()
    for p in ("fontspec", "hyperref", "longtable", "needspace", "fvextra", "xurl"):
        assert p in pkgs


def test_template_text_layer_and_code_wraps():
    tpl = (B.KIT / "template.tex").read_text(encoding="utf-8")
    # copied and searched text gets the source characters (";" not U+037E)
    assert re.search(r"^\\XeTeXgenerateactualtext=1", tpl, re.M)
    # a wrapped code line is marked, so it is not mistaken for a command of its own
    m = re.search(r"breaksymbolleft=\{([^}]*\})", tpl)
    assert m and "hookrightarrow" in m.group(1)


# ------------------------------------------------------------- the Lua filter

MINI = """# A small report

Technical report for a test [Brier 1950].

Author: A. B. Cee

Affiliation: Somewhere

Code: https://example.org/code

Keywords: alpha, beta

## Abstract

The abstract quotes 0.123.

## 1 Introduction

See §2.1, App A.1, Figure 1, Table 1 and [Brier 1950]. Also §9.9 and App B.1.

## 2 Method

### 2.1 Part

*Table 1: things.*

| name | value |
|---|---|
| `experiments/some_long_script_name.py` | -0.5 |
| plain | 0.25 |

![Figure 1: alt text](fig/item_gap.png)

*Figure 1.* A caption with 0.42 in it.

## References

- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1): 1–3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

## Appendix A: the first appendix

### A.1 One

Back to §1 and Table 1.

| a | b |
|---|---|
| 1 | 2 |

{AFTER}
"""
#: the paragraph after the last table: the filter's report names the table by its
#: first 50 characters, which a cut by bytes would split inside an "é"
AFTER = "x" + "é" * 60
MINI = MINI.replace("{AFTER}", AFTER)


@pytest.fixture
def mini(tmp_path):
    src = tmp_path / "src"
    (src / "fig").mkdir(parents=True)
    shutil.copyfile(FIG / "item_gap.png", src / "fig" / "item_gap.png")
    shutil.copyfile(FIG / "item_gap.svg", src / "fig" / "item_gap.svg")
    draft = src / "draft.md"
    draft.write_text(MINI, encoding="utf-8")
    return draft


@needs_pandoc
def test_filter_on_a_small_draft(mini, tmp_path):
    r = B.build(mini, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    tex = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[1]
    flt = r["filter"]
    assert sorted((u["kind"], u["text"]) for u in flt["unresolved"]) == [
        ("appendix", "App B.1"), ("section", "§9.9")]
    assert flt["warnings"] == []
    assert flt["links"]["section"] == 2 and flt["links"]["appendix"] == 1
    assert flt["links"]["figure"] == 1 and flt["links"]["table"] == 2
    assert flt["links"]["citation"] == 2 and flt["links"]["external"] == 1
    assert flt["references"][0]["id"] == "ref-brier-1950"
    # title block, numbered headings with stable ids, the appendix after a page break
    assert "{\\large\\bfseries A. B. Cee\\par}" in tex
    assert "\\section{\\secnum{1}Introduction}\\label{sec-1}" in tex
    assert "\\subsection{\\secnum{A.1}One}\\label{app-A.1}" in tex
    assert tex.index("\\reportappendixstart") < tex.index("\\label{app-A}")
    assert tex.index("\\label{references}") < tex.index("\\reportappendixstart")
    # the figure as a float with the draft's caption; the short table kept on one page
    assert "\\label{fig-1}" in tex and "\\reportfig{3.410in}{fig/item_gap.png}" in tex
    assert "\\reportcaption{Figure 1.}{A caption with 0.42 in it.}" in tex
    assert "\\reportneed{" in tex and "\\needspace{" not in tex
    assert flt["tables"][0]["kept_together"] is True
    assert "\\begin{tabular}" in tex and "−0.5" in tex   # a negative number gets a minus sign
    # the snippet that names a table is cut by character, never inside a UTF-8 sequence
    assert flt["tables"][-1]["where"] == "before: " + AFTER[:50]
    # the "Keywords:" line is metadata only: in hyperref's setup, not in the body
    preamble = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[0]
    assert "pdfkeywords={alpha, beta}" in preamble and "alpha" not in tex
    assert "https://doi.org/10.1175/1520-0493(1950)078\\%3C0001:VOFEIT\\%3E2.0.CO;2" in tex


@needs_pandoc
def test_filter_on_the_draft_resolves_everything(tmp_path):
    r = B.build(DRAFT, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    flt = r["filter"]
    assert flt["unresolved"] == [] and flt["warnings"] == []
    assert r["figure_placement"] == [] and r["pandoc_stderr"] == []
    tex = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[1]
    for label in B.figures_md_placement(FIGURES_MD.read_text(encoding="utf-8")):
        assert f"\\label{{fig-{label}}}" in tex, label
    assert tex.count("\\reportappendixstart") == 1
    assert EMAIL_RE.findall(tex.split("\\section", 1)[0]) in ([], ["nikitapol@fbb.msu.ru"] * 2,
                                                             ["nikitapol@fbb.msu.ru"])


# ------------------------------------------------------------------- the PDF

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("report")
    return B.build(DRAFT, out, figures="png", epoch=1_700_000_000, quiet=True)


@needs_latex
def test_report_pdf_builds_clean(built):
    r = built
    assert Path(r["pdf"]).exists() and r["pdf_within_limit"]
    log = r["log"]
    assert [o for o in log["overfull"] if o["pt"] > B.OVERFULL_LIMIT_PT] == []
    assert log["missing_glyphs"] == [] and log["undefined_refs"] == []
    assert log["pdf_string_tokens"] == 0
    # --strict refuses only the placeholders and TODO markers left for the team
    assert all(p.startswith(("title block placeholders", f"{len(r['todo_markers'])} TODO markers"))
               for p in B.problems(r))
    tc = r["text_check"]
    if "skipped" not in tc:
        assert tc["numbers_missing"] == {} and tc["words_missing"] == {}


@needs_latex
def test_report_pdf_layout(built):
    r = built
    pages, lp = r["pages"], r["label_pages"]
    assert pages["main_text"][0] == 1 and pages["appendices"][1] == pages["total"]
    assert pages["references"][0] <= pages["main_text"][1] < pages["appendices"][0]
    assert lp["sec-1"] == 1 and lp["references"] < lp["report:appendix-start"]
    labels = B.figures_md_placement(FIGURES_MD.read_text(encoding="utf-8"))
    main = [lab for lab in labels if lab.isdigit()]
    for lab in labels:
        page = lp[f"fig-{lab}"]
        if lab in main:
            assert page < lp["references"], lab
        else:
            assert page >= lp["report:appendix-start"], lab
    assert [lp[f"fig-{lab}"] for lab in main] == sorted(lp[f"fig-{lab}"] for lab in main)


@needs_latex
def test_report_pdf_metadata_names_the_author_only(built):
    pdf = Path(built["pdf"]).read_bytes()
    assert b"%PDF-" == pdf[:5]
    pypdf = pytest.importorskip("pypdf")
    info = pypdf.PdfReader(built["pdf"]).metadata
    assert info.author == B.front_matter(DRAFT.read_text(encoding="utf-8"))["author"]
    assert info.title.startswith("Calibrating")
    assert info.get("/Keywords") == B.front_matter(DRAFT.read_text(encoding="utf-8"))["keywords"]
    assert not any(EMAIL_RE.search(str(v)) for v in info.values())


@needs_latex
def test_build_is_deterministic(mini, tmp_path):
    a = B.build(mini, tmp_path / "a", figures="png", epoch=1_700_000_000, quiet=True)
    b = B.build(mini, tmp_path / "b", figures="png", epoch=1_700_000_000, quiet=True)
    assert a["pdf_sha256"] == b["pdf_sha256"]
    assert hashlib.sha256(Path(a["pdf"]).read_bytes()).hexdigest() == a["pdf_sha256"]
    assert a["pages"]["total"] >= 2 and a["pages"]["appendices"][0] == a["pages"]["total"]
