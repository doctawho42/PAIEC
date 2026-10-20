"""tools/build_report_pdf.py: docs/report/draft.md builds into the report PDF.

The pure-Python parts (title block, LaTeX log and .aux parsing, the text check, figure
placement against docs/report/figures.md, the page measure, the NeurIPS style check)
run everywhere. The Lua filter tests need pandoc, and the PDF builds need pandoc, XeLaTeX
with the template's packages and the NeurIPS 2026 style file (tools/report_pdf/
neurips_2026.sty, not committed: the build names its download); they are skipped when
those are missing (looked up as the build does: PATH, then /opt/homebrew/bin,
/opt/local/bin, /usr/local/bin and /Library/TeX/texbin; PANDOC and XELATEX override).
No data download and no network are needed.
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
STYLE = B.style_status()

needs_pandoc = pytest.mark.skipif(not PANDOC, reason="pandoc not found (set PANDOC)")
needs_latex = pytest.mark.skipif(
    not (PANDOC and XELATEX and not LACKING and STYLE["ok"]),
    reason="pandoc not found (set PANDOC)" if not PANDOC
    else "xelatex not found (set XELATEX)" if not XELATEX
    else f"the TeX installation lacks {', '.join(LACKING or [])}" if LACKING
    else B.style_instructions(STYLE))

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


#: decision (a) of 2026-10-20: the NeurIPS acknowledgments section's one paragraph
ACK_SENTENCE = ("This work received no specific funding. "
                "The author declares no competing interests.")


def test_draft_declares_funding_and_competing_interests():
    """The NeurIPS "Acknowledgments and Disclosure of Funding" section, unnumbered, after
    the main text and right before References, with exactly the author's sentence."""
    text = DRAFT.read_text(encoding="utf-8")
    parts = text.split(f"\n## {B.ACK_TITLE}\n", 1)
    assert len(parts) == 2 and text.count(f"## {B.ACK_TITLE}") == 1
    body, after = parts[1].split("\n## ", 1)
    assert after.startswith("References\n")
    assert [b.strip() for b in body.split("\n\n") if b.strip() not in ("", "---")] == [ACK_SENTENCE]
    # after every numbered section of the main text, before every appendix
    assert "\n## Appendix A:" not in parts[0] and "\n## 8 " in parts[0]


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
        assert 1.0 < B.svg_width_in(FIG / f"{n}.svg") <= 5.5, n   # within the 5.5in text width


def test_parse_log():
    log = "\n".join([
        "Overfull \\hbox (4.5pt too wide) in paragraph at lines 10--12",
        "[]\\TU/X/m/n/10 some text",
        "Overfull \\hbox (0.8pt too wide) in alignment at lines 20--30",
        "[] []",
        "Overfull \\hbox (24.8pt too wide) detected at line 4250",
        "[] [][]",
        "Underfull \\hbox (badness 10000) in paragraph at lines 3--4",
        "Missing character: There is no Ж in font TeX Gyre Termes!",
        "LaTeX Warning: Reference `sec-9' on page 3 undefined on input line 7.",
        "LaTeX Warning: Float too large for page by 4.0pt on input line 9.",
        "Output written on report.pdf (12 pages).",
    ])
    p = B.parse_log(log)
    assert [o["pt"] for o in p["overfull"]] == [4.5, 0.8, 24.8]
    assert p["overfull"][1]["where"] == "alignment" and p["overfull"][0]["lines"] == "10--12"
    # a box (a display, or a table kept on one page) gives one line, which is kept
    assert p["overfull"][2]["where"] == "box" and p["overfull"][2]["lines"] == "4250"
    assert p["underfull"] == 1 and p["missing_glyphs"] == ["Ж"]
    assert p["undefined_refs"] == ["sec-9"] and p["pages"] == 12
    assert p["other_warnings"] == ["LaTeX Warning: Float too large for page by 4.0pt on input line 9."]
    assert p["rerun"] is False
    assert B.parse_log("LaTeX Warning: Label(s) may have changed. Rerun to get it right.")["rerun"]


#: the text block of the NeurIPS style: its top 10in above the foot of a letter page, 9in high
GEOMETRY = (722.7, 650.43)


def _y(frac: float) -> int:
    """The \\pdflastypos (sp from the foot of the page) of a point frac down the text block."""
    return round((GEOMETRY[0] - frac * GEOMETRY[1]) * B.SP_PER_PT)


def test_aux_pages_and_page_summary():
    aux = ("\\newlabel{sec-1}{{}{1}{}{section*.1}{}}\n"
           "\\newlabel{references}{{}{8}{}{section*.40}{}}\n"
           "\\newlabel{fig-1}{{}{3}{}{figure.1}{}}\n"
           "\\newlabel{fig-2}{{}{9}{}{figure.2}{}}\n"
           "\\newlabel{report:appendix-start}{{}{11}{}{section*.41}{}}\n"
           "\\newlabel{report:end}{{}{74}{}{}{}}\n"
           f"\\reportposition{{sec-1}}{{1}}{{{_y(0.6)}}}\n"
           f"\\reportposition{{report:main-end}}{{8}}{{{_y(0.25)}}}\n"
           f"\\reportposition{{references}}{{8}}{{{_y(0.25)}}}\n"
           "\\reportgeometry{722.7pt}{650.43pt}\n")
    labels = B.aux_pages(aux)
    assert labels == {"sec-1": 1, "references": 8, "fig-1": 3, "fig-2": 9,
                      "report:appendix-start": 11, "report:end": 74}
    positions, geometry = B.aux_positions(aux)
    assert geometry == GEOMETRY and positions["report:main-end"][0] == 8
    s = B.page_summary(labels, 74, positions, geometry, main_figures=["1"])
    # the main text ends a quarter down page 8, where References begins
    assert s["main_text"] == [1, 8] and s["main_text_pages"] == 8
    assert s["main_text_measured"] == 7.25 and s["main_text_basis"] == "end mark"
    assert s["references"] == [8, 10] and s["appendices"] == [11, 74]
    assert s["appendix_pages"] == 64 and s["main_text_limit"] == B.MAX_MAIN_PAGES == 8
    # a figure of the main text that floats past the end of its text counts
    s = B.page_summary(labels, 74, positions, geometry, main_figures=["1", "2"])
    assert s["main_text_pages"] == 9 and s["main_text_basis"] == "end mark and figures"
    assert s["main_text_ends_before"] == "the end of the document"   # main_end not given
    # without the end mark (an older .tex): the page before References
    s = B.page_summary(labels, 74)
    assert s["main_text"] == [1, 7] and s["references"] == [8, 10]
    assert "acknowledgments" not in s


def test_page_summary_ends_the_main_text_at_the_acknowledgments():
    """report.lua sets the end mark before the "Acknowledgments and Disclosure of Funding"
    heading, which the NeurIPS instructions do not count as content pages: the main text
    is measured to that mark, and build.json names the heading and the section's page."""
    labels = {"references": 9, B.ACK_LABEL: 8, "report:appendix-start": 11, "report:end": 40}
    positions = {"report:main-end": (8, _y(0.8) / B.SP_PER_PT),
                 "acknowledgments-and-disclosure-of-funding": (8, _y(0.8) / B.SP_PER_PT),
                 "references": (9, _y(0.0) / B.SP_PER_PT)}
    s = B.page_summary(labels, 40, positions, GEOMETRY, main_end=B.ACK_TITLE)
    assert s["main_text"] == [1, 8] and s["main_text_measured"] == 7.8
    assert s["main_text_ends_before"] == B.ACK_TITLE and s["acknowledgments"] == 8
    assert s["references"] == [9, 10] and s["main_text_basis"] == "end mark"
    # an acknowledgments section that pushes References to the next page adds nothing
    # to the main text; without the end mark, the page before the acknowledgments
    s = B.page_summary(dict(labels, **{B.ACK_LABEL: 9}), 40)
    assert s["main_text"] == [1, 8] and s["main_text_basis"] == "page before the acknowledgments"


def test_section_pages_and_words():
    headings = [
        {"id": "sec-1", "level": 2, "text": "1 Introduction", "words": 100, "main": True},
        {"id": "sec-1.1", "level": 3, "text": "1.1 Task", "words": 300, "main": True},
        {"id": "sec-1.2", "level": 3, "text": "1.2 Why", "words": 200, "main": True},
        {"id": "sec-2", "level": 2, "text": "2 Data", "words": 400, "main": True},
        {"id": "references", "level": 2, "text": "References", "words": 50, "main": False},
        {"id": "app-A", "level": 2, "text": "Appendix A: data", "words": 10, "main": False},
    ]
    pos = {"sec-1": (1, _y(0.5) / B.SP_PER_PT), "sec-1.1": (1, _y(0.6) / B.SP_PER_PT),
           "sec-1.2": (2, _y(0.5) / B.SP_PER_PT), "sec-2": (3, _y(0.0) / B.SP_PER_PT),
           "report:main-end": (4, _y(0.75) / B.SP_PER_PT),
           "references": (4, _y(0.75) / B.SP_PER_PT), "app-A": (6, _y(0.0) / B.SP_PER_PT)}
    rows = {r["id"]: r for r in B.section_pages(headings, pos, GEOMETRY, 9)}
    assert rows["sec-1"]["words"] == 600 and rows["sec-1.1"]["words"] == 300
    assert rows["sec-1"]["pages"] == 1.5 and rows["sec-1"]["start_page"] == 1
    assert rows["sec-1.1"]["pages"] == 0.9 and rows["sec-1.2"]["pages"] == 0.5
    assert rows["sec-2"]["pages"] == 1.75          # to the main text's end mark
    assert rows["references"]["pages"] == 1.25 and not rows["references"]["main"]
    assert rows["app-A"]["pages"] == 4.0           # to the end of the document


def test_strict_holds_the_main_text_to_eight_pages():
    """The competition: "approximately 4-8 pages of main text, excluding references and
    appendices", in the NeurIPS template."""
    def report(n):
        return {"pages": {"main_text_pages": n, "main_text_measured": n - 0.4}}

    assert B.problems(report(8)) == [] and B.problems(report(5)) == []
    bad = B.problems(report(9))
    assert len(bad) == 1 and bad[0].startswith("main text is 9 pages")
    assert "acknowledgments, references and appendices excluded" in bad[0]
    assert B.problems(report(9), max_main_pages=9) == []


def test_strict_wants_the_acknowledgments_right_after_the_main_text():
    """The page measure ends the main text at the acknowledgments heading only where the
    NeurIPS template puts the section: after the main text, before References."""
    heads = [{"id": "sec-1", "level": 2, "text": "1 Introduction"},
             {"id": "ack", "level": 2, "text": B.ACK_TITLE},
             {"id": "references", "level": 2, "text": "References"}]
    assert B.problems({"filter": {"headings": heads, "main_end": B.ACK_TITLE}}) == []
    # after References (the main text ended there), or a draft whose filter did not end it
    for main_end in ("References", None):
        bad = B.problems({"filter": {"headings": heads, "main_end": main_end}})
        assert len(bad) == 1 and B.ACK_TITLE in bad[0] and "between the main text" in bad[0]
    # inside the main text: the filter ends the main text at the heading all the same, and
    # the sections after it would silently stop counting
    lim = {"id": "sec-2", "level": 2, "text": "2 Limitations"}
    code = {"id": "code-data-and-conduct", "level": 2, "text": "Code, data and conduct"}
    for follows in ([lim, code], [code]):
        bad = B.problems({"filter": {"headings": heads[:2] + follows + heads[2:],
                                     "main_end": B.ACK_TITLE}})
        assert (len(bad) == 1 and f'followed by "{follows[0]["text"]}"' in bad[0]
                and "between the main text" in bad[0])
    # an appendix right after it is fine (a draft without References)
    app = [heads[0], heads[1], {"id": "app-A", "level": 2, "text": "Appendix A: data"}]
    assert B.problems({"filter": {"headings": app, "main_end": B.ACK_TITLE}}) == []
    # a draft without the section: nothing to check
    assert B.problems({"filter": {"headings": heads[::2], "main_end": "References"}}) == []


PDFFONTS = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ZBVPBB+TimesNewRomanPS-BoldMT        CID TrueType      Identity-H       yes yes yes      4  0
LOFTVK+STIXTwoMath-Regular-Identity-H CID Type 0C       Identity-H       yes yes yes      6  0
[none]                               Type 3            Custom           yes no  no      12  0
Helvetica                            Type 1            Custom           no  no  no      14  0
"""


def test_font_check_refuses_type3_and_unembedded_fonts():
    """The NeurIPS instructions: "Type 1 or Embedded TrueType fonts" only."""
    fonts = B.parse_pdffonts(PDFFONTS)
    assert [f["type"] for f in fonts] == ["CID TrueType", "CID Type 0C", "Type 3", "Type 1"]
    assert fonts[0]["name"] == "ZBVPBB+TimesNewRomanPS-BoldMT" and fonts[0]["embedded"]
    assert not fonts[3]["embedded"]
    assert B.bad_fonts(fonts) == ["[none] (Type 3)", "Helvetica (Type 1, not embedded)"]
    assert B.bad_fonts(fonts[:2]) == []
    bad = B.problems({"fonts": fonts})
    assert len(bad) == 1 and "Type 3 or not embedded" in bad[0]
    assert B.problems({"fonts": {"skipped": "pdffonts not found"}}) == []


def test_style_check_names_the_download_and_the_pin(tmp_path):
    """The NeurIPS style is not committed (no license): a missing or different copy stops
    the build with the official download and the pinned sha256."""
    missing = tmp_path / B.STYLE_NAME
    with pytest.raises(B.BuildError) as e:
        B.check_style(missing)
    msg = str(e.value)
    assert "is missing" in msg and B.STYLE_URL in msg and B.STYLE_SHA256 in msg
    assert B.STYLE_ZIP_SHA256 in msg and "not committed" in msg
    missing.write_text("% neurips_2026.sty, edited\n", encoding="utf-8")
    st = B.style_status(missing)
    assert st["present"] and not st["ok"] and st["sha256"] != B.STYLE_SHA256
    with pytest.raises(B.BuildError) as e:
        B.check_style(missing)
    assert st["sha256"] in str(e.value) and B.STYLE_URL in str(e.value)
    # the pins: the official zip of 2026-10-20 and the style in it
    assert B.STYLE_URL == ("https://media.neurips.cc/Conferences/NeurIPS2026/"
                           "Formatting_Instructions_For_NeurIPS_2026.zip")
    assert B.STYLE_SHA256 == "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a"
    if STYLE["present"]:
        assert B.check_style()["ok"], B.style_instructions(STYLE)


def test_style_file_is_not_committed():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "tools/report_pdf/neurips_2026.sty" in gi
    if shutil.which("git"):
        import subprocess
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "tools/report_pdf"],
                                 capture_output=True, text=True).stdout.split()
        assert "tools/report_pdf/neurips_2026.sty" not in tracked


@needs_pandoc
def test_build_stops_without_the_style(mini, tmp_path, monkeypatch):
    kit = tmp_path / "kit"
    shutil.copytree(B.KIT, kit, ignore=shutil.ignore_patterns(B.STYLE_NAME))
    monkeypatch.setattr(B, "KIT", kit)
    with pytest.raises(B.BuildError) as e:
        B.build(mini, tmp_path / "out", figures="png", epoch=0, quiet=True)
    assert B.STYLE_URL in str(e.value) and B.STYLE_SHA256 in str(e.value)
    # --tex-only needs no style: it records that the style is missing and goes on
    r = B.build(mini, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    assert r["style"]["present"] is False and r["style"]["ok"] is False


def test_template_uses_the_neurips_style_unchanged():
    """The style's geometry and font sizes are the template's: the template sets neither
    (the NeurIPS instructions forbid changing them)."""
    tpl = (B.KIT / "template.tex").read_text(encoding="utf-8")
    code = "\n".join(ln.split("%", 1)[0] for ln in tpl.splitlines())
    assert re.search(r"^\\documentclass\{article\}$", code, re.M)
    assert "\\usepackage[sglblindworkshop,final]{neurips_2026}" in code
    assert "\\workshoptitle{Predictive AI Evaluation Competition (PAIEC)}" in code
    for banned in ("{geometry}", "\\newgeometry", "titlesec", "\\setlength{\\textwidth}",
                   "\\setlength{\\textheight}", "\\addtolength",
                   "\\fontsize", "\\renewcommand{\\normalsize}", "\\linespread",
                   "\\topfraction", "\\floatpagefraction", "\\setlist", "\\parindent"):
        assert banned not in code, banned
    # the paragraph skip is the style's 5.5pt; only pandoc's tight lists drop it, locally
    assert re.findall(r"\\setlength\{\\parskip\}", code) == ["\\setlength{\\parskip}"]
    assert ("\\providecommand{\\tightlist}{\\setlength{\\itemsep}{0pt}"
            "\\setlength{\\parskip}{0pt}}") in code
    # the text face is the style's ptm: a Times-like OpenType face under that NFSS name
    assert len(re.findall(r"\\setmainfont\{[^}]+\}\[NFSSFamily=ptm", code)) == 3
    assert code.index("{fontspec}") < code.index("{neurips_2026}") < code.index("{unicode-math}")


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
    # title block: the NeurIPS title and author block in the preamble; the subtitle and
    # the code line a footnote to the title
    preamble = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[0]
    assert "\\title{A small report\\thanks{Technical report for a test" in preamble
    assert "Code: \\url{https://example.org/code}}}" in preamble
    assert "\\author{%\n  A. B. Cee \\\\\n  Somewhere \\\\\n}" in preamble
    assert "\\maketitle" in tex and "\\begin{abstract}" in tex
    # numbered headings with stable ids and page marks, the appendix after a page break
    assert ("\\reportpos{sec-1}\\section{\\secnum{1}Introduction}\\label{sec-1}\\nobreak"
            in tex)
    assert "\\subsection{\\secnum{A.1}One}\\label{app-A.1}" in tex
    assert tex.index("\\reportappendixstart") < tex.index("\\label{app-A}")
    assert tex.index("\\label{references}") < tex.index("\\reportappendixstart")
    # the main text ends before References: its end mark, then References' own mark
    assert (tex.index("\\reportpos{report:main-end}") < tex.index("\\reportpos{references}")
            < tex.index("\\label{references}"))
    assert tex.count("\\reportpos{report:main-end}") == 1
    heads = {h["id"]: h for h in flt["headings"]}
    assert heads["sec-1"]["main"] and heads["sec-2.1"]["main"] and not heads["references"]["main"]
    assert not heads["app-A.1"]["main"] and heads["sec-1"]["words"] == 16
    assert flt["figures"][0]["main"] is True
    # the figure as a float with the draft's caption; the short table kept on one page
    assert "\\label{fig-1}" in tex and "\\reportfig{3.410in}{fig/item_gap.png}" in tex
    assert "\\reportcaption{Figure 1.}{A caption with 0.42 in it.}" in tex
    assert "\\reportneed{" in tex and "\\needspace{" not in tex
    assert flt["tables"][0]["kept_together"] is True
    assert "\\begin{tabular}" in tex and "−0.5" in tex   # a negative number gets a minus sign
    # the snippet that names a table is cut by character, never inside a UTF-8 sequence
    assert flt["tables"][-1]["where"] == "before: " + AFTER[:50]
    # the "Keywords:" line is metadata only: in hyperref's setup, not in the body
    assert "pdfkeywords={alpha, beta}" in preamble and "alpha" not in tex
    assert "https://doi.org/10.1175/1520-0493(1950)078\\%3C0001:VOFEIT\\%3E2.0.CO;2" in tex


@needs_pandoc
def test_filter_links_appendix_tables(mini, tmp_path):
    """"Table A.1a" links to its appendix table's caption; the caption does not link
    to itself, its anchor sits inside the caption's paragraph (no block of its own),
    a label without a caption is reported, and a possessive links as well."""
    text = mini.read_text(encoding="utf-8")
    text = text.replace("Also §9.9 and App B.1.",
                        "Also §9.9, App B.1, App A.1’s one, Table A.1a's and Table A.9z.")
    text = text.replace("Back to §1 and Table 1.\n\n",
                        "Back to §1 and Table 1.\n\n*Table A.1a: a Table A.1a of things.*\n\n")
    mini.write_text(text, encoding="utf-8")
    r = B.build(mini, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    tex = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[1]
    flt = r["filter"]
    assert ("table", "Table A.9z") in [(u["kind"], u["text"]) for u in flt["unresolved"]]
    assert flt["links"]["table"] == 3   # Table 1 twice, Table A.1a once
    # possessives link whatever the locale (Lua's %w is locale-dependent)
    assert flt["links"]["appendix"] == 2
    assert "\\hyperref[tab-A.1a]{Table A.1a}'s" in tex
    assert "\\hyperref[app-A.1]{App A.1}'s one" in tex
    assert "\\phantomsection\\label{tab-A.1a}\\emph{Table A.1a: a Table A.1a of things.}" in tex
    assert tex.count("tab-A.1a") == 2
    # the caption keeps the lead-in's space request before the table, as without the anchor
    cap = tex.index("\\label{tab-A.1a}")
    assert tex.rindex("\\reportneed{", 0, cap) > tex.index("Back to")


ACK_SECTION = f"## {B.ACK_TITLE}\n\n{ACK_SENTENCE}\n\n---\n\n"
ACK_ID = "acknowledgments-and-disclosure-of-funding"


def _with_ack(draft: Path) -> None:
    text = draft.read_text(encoding="utf-8")
    assert text.count("## References\n") == 1
    draft.write_text(text.replace("## References\n", ACK_SECTION + "## References\n"),
                     encoding="utf-8")


@needs_pandoc
@pytest.mark.parametrize("with_ack", [True, False], ids=["with-ack", "without-ack"])
def test_filter_ends_the_main_text_at_the_acknowledgments(mini, tmp_path, with_ack):
    """With the NeurIPS "Acknowledgments and Disclosure of Funding" section, the main
    text's end mark comes before its heading, and the section is set in the style's ack
    environment (which prints the heading, unnumbered), closed before References;
    without it, the end mark comes before References, as before."""
    if with_ack:
        _with_ack(mini)
    r = B.build(mini, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    tex = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[1]
    flt = r["filter"]
    assert flt["warnings"] == [] and len(flt["unresolved"]) == 2
    assert tex.count("\\reportpos{report:main-end}") == 1
    end = tex.index("\\reportpos{report:main-end}")
    refs = tex.index("\\reportpos{references}")
    heads = {h["id"]: h for h in flt["headings"]}
    assert heads["sec-2.1"]["main"] and not heads["references"]["main"]
    if with_ack:
        assert flt["main_end"] == B.ACK_TITLE
        begin, close = tex.index("\\begin{ack}"), tex.index("\\end{ack}")
        assert tex.count("\\begin{ack}") == tex.count("\\end{ack}") == 1
        assert end < tex.index(f"\\reportpos{{{ACK_ID}}}") < begin < close < refs
        # end mark, the heading's own mark, then the environment: nothing between
        assert tex[end:begin].split() == ["\\reportpos{report:main-end}", f"\\reportpos{{{ACK_ID}}}"]
        # the heading is the style's (\acksection), with a bookmark and a page label
        assert ("\\begin{ack}\\phantomsection\\pdfbookmark[1]{" + B.ACK_TITLE + "}{" + ACK_ID
                + "}\\label{report:ack}\\nobreak") in tex
        assert "\\section{Acknowledgments" not in tex and tex.count(B.ACK_TITLE) == 1
        assert ACK_SENTENCE in tex[begin:close]
        assert not heads[ACK_ID]["main"] and heads[ACK_ID]["words"] == len(ACK_SENTENCE.split())
    else:
        assert flt["main_end"] == "References" and "{ack}" not in tex and ACK_ID not in heads
        assert end < refs


@needs_pandoc
@pytest.mark.parametrize("locale", ["C", "C.UTF-8", "en_US.UTF-8"])
def test_filter_links_a_possessive_without_a_letter_suffix(mini, tmp_path, monkeypatch, locale):
    """"Table A.6's" (smart quotes make it "A.6’s") links to Table A.6 under any locale:
    the label pattern uses ASCII classes, since under a UTF-8 locale Lua's %l took the
    first byte of "’" (0xE2) for a letter and made the label "A.6\\xE2"."""
    monkeypatch.setenv("LC_ALL", locale)
    text = mini.read_text(encoding="utf-8")
    text = text.replace("Also §9.9 and App B.1.",
                        "Also §9.9, App B.1, Table A.6's one, Table A.6’s two and Table A.6a's.")
    text = text.replace("Back to §1 and Table 1.\n\n",
                        "Back to §1 and Table 1.\n\n*Table A.6: things.*\n\n*Table A.6a: more.*\n\n")
    mini.write_text(text, encoding="utf-8")
    r = B.build(mini, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    tex = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[1]
    flt = r["filter"]
    assert sorted(u["text"] for u in flt["unresolved"]) == ["App B.1", "§9.9"]
    assert flt["links"]["table"] == 5     # Table 1 twice, A.6 twice, A.6a once
    assert tex.count("\\hyperref[tab-A.6]{Table A.6}'s") == 2
    assert "\\hyperref[tab-A.6a]{Table A.6a}'s" in tex
    assert "\\label{tab-A.6}\\emph{Table A.6: things.}" in tex


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
    assert tex.count("\\reportpos{report:main-end}") == 1
    # the NeurIPS acknowledgments section ends the main text (decision (a) of 2026-10-20)
    assert flt["main_end"] == B.ACK_TITLE and tex.count("\\begin{ack}") == 1
    assert (tex.index("\\reportpos{report:main-end}") < tex.index("\\begin{ack}")
            < tex.index(ACK_SENTENCE) < tex.index("\\end{ack}") < tex.index("\\reportpos{references}"))
    # the team's e-mail, in the author block only (as the link and as its text)
    whole = Path(r["tex"]).read_text(encoding="utf-8")
    author = whole.split("\\author{", 1)[1].split("\n}\n", 1)[0]
    assert EMAIL_RE.findall(author) == ["nikitapol@fbb.msu.ru"] * 2
    assert EMAIL_RE.findall(whole.replace(author, "")) == []


# ------------------------------------------------------------------ TeX math

MATH = r"""# A small report with math

Technical report for a test.

Author: A. B. Cee

Code: https://example.org/code

Keywords: alpha, beta

## Abstract

The abstract quotes 0.123 and $B_{31} = 0.1725$.

## 1 Introduction

The level prior is $\mu_b \sim \mathcal{N}(\mu_0, \sigma_\mu^2)$ with $\mu_0 = -2.5$, $\sigma_\mu = 2.5$
and $\gamma = 0.5$; a gain of $\Delta = -0.0419$ at $B_{31}$, and $\mathbb{E}[p(1-p)]$ near 0.128.
See Table 1, Figure 1 and App A.1. It costs $5 and $6, which stay text.

$$
\mathrm{ALC} = 0.1\,B_0 + 0.2\,(B_1 + B_3 + B_7 + B_{15}) + 0.1\,B_{31}
$$

The response model, with the item residual integrated:

$$
\eta_{si} = \mu_b + \theta_s + \delta_{sb} - g_i - e_i, \qquad
\hat p_{si} = c_i + (1 - c_i - \varepsilon)\,\mathbb{E}\big[\operatorname{logit}^{-1}(\eta_{si})\big]
$$

- *the level $\mu_b$ in emphasis*, and **a gain of $-0.042$ in bold**;
- a list item with $p ≤ 0.05$ in math, and p ≤ 0.06 → text.

*Table 1: the level prior $\mu_b$.*

| quantity | symbol | value |
|---|---|---|
| level centre | $\mu_0$ | $-2.5$ |
| level sd | $\sigma_\mu$ | 2.5 |
| item variance | $\sigma_d^2 + \sigma_g^2$ | $2.671^2 + 1.542^2$ |
| harness offset | $\operatorname{cap}(o) = 4\tanh(o/4)$ | -0.5 |

Source: a note with $\pm 0.01$ in it.

![Figure 1: alt text](fig/item_gap.png)

*Figure 1.* A caption with $r = 0.3$ and $\sqrt{1 - r^2}$ in it, 0.42.

## Appendix A: the first appendix

### A.1 One

A thin space in $571\,921$ responses, $10^{-4}$, $4.8\%$ of items and $\tfrac{1}{2}$.

$$
\begin{aligned}
a &= 1.25 \\[2pt]
b &= \text{rest of it} % a comment 99, never printed
\end{aligned}
$$

$$
\begin{align}
x &= 7.5 \\
y &\le 8.5
\end{align}
$$
"""


@pytest.fixture
def mini_math(tmp_path):
    src = tmp_path / "src_math"
    (src / "fig").mkdir(parents=True)
    shutil.copyfile(FIG / "item_gap.png", src / "fig" / "item_gap.png")
    shutil.copyfile(FIG / "item_gap.svg", src / "fig" / "item_gap.svg")
    draft = src / "draft.md"
    draft.write_text(MATH, encoding="utf-8")
    return draft


def test_defaults_read_tex_math():
    """The draft's $...$ and $$...$$ are math, as GitHub renders them."""
    reader = B.pandoc_reader()
    assert reader.startswith("gfm")
    assert "-tex_math_dollars" not in reader and "-tex_math_gfm" not in reader


def test_template_loads_unicode_math_after_amsmath_and_fontspec():
    pkgs = B.latex_packages()
    assert "amsmath" in pkgs and "unicode-math" in pkgs
    assert pkgs.index("fontspec") < pkgs.index("unicode-math")
    assert pkgs.index("amsmath") < pkgs.index("unicode-math")
    tpl = (B.KIT / "template.tex").read_text(encoding="utf-8")
    assert re.search(r"^\s*\\setmathfont\{STIX Two Math\}", tpl, re.M)
    # pandoc's template language reads a dollar sign as a variable
    body = [ln for ln in tpl.splitlines() if ln.lstrip().startswith("%")]
    assert not any("$" in ln for ln in body)
    # a symbol the text fallback takes from the math font is the math symbol in math
    defs = re.findall(r"^\s*\\newunicodechar\{(.)\}\{(.*)\}$", tpl, re.M)
    assert len(defs) == 6 and all(d.startswith("\\reportsymbol{") for _, d in defs)
    assert re.search(r"\\protected\\def\\reportsymbol#1#2\{\\ifmmode#2\\else", tpl)
    # amsmath's \boldsymbol has no bold math version in an OpenType math font, so the
    # draft's \boldsymbol\Sigma goes through unicode-math's \symbf, after unicode-math
    m = re.search(r"^\\AtBeginDocument\{\\renewcommand\{\\boldsymbol\}\[1\]\{\\symbf\{#1\}\}\}$",
                  tpl, re.M)
    assert m and m.start() > tpl.index("{unicode-math}")


def test_math_text_keeps_numbers_and_printed_words():
    def nums(tex):
        return B.NUMBER_RE.findall(B._normalise(B.math_text(tex)))

    def words(tex):
        return B.WORD_RE.findall(B._normalise(B.math_text(tex)))

    assert nums(r"\mu_0 = -2.5,\ \sigma_\mu = 2.5,\ \gamma = 0.5") == ["0", "2.5", "2.5", "0.5"]
    assert words(r"\mu_0 = -2.5,\ \sigma_\mu = 2.5") == []      # command names are not words
    tex = r"\mathrm{ALC} = 0.1\,B_0 + 0.2\,(B_1 + B_{15}) + 0.1\,B_{31}"
    assert nums(tex) == ["0.1", "0", "0.2", "1", "15", "0.1", "31"]
    assert words(tex) == ["ALC", "B", "B", "B", "B"]
    assert words(r"\operatorname{logit}^{-1}(\eta) - \text{hier-ship}") == ["logit", "hiership"]
    assert nums(r"\tfrac{1}{2}") == ["1", "2"]                  # never "12"
    # spacing, labels and environment names print nothing; \% prints
    tex = r"\begin{aligned} x &= 1{,}000 \\ \hspace{2pt} y &= 4.8\% \end{aligned}"
    assert nums(tex) == ["1", "000", "4.8"] and words(tex) == ["x", "y"]
    assert "%" in B.math_text(tex)
    # nor do comments, column specs, a row's extra space, bare dimensions and label keys
    tex = ("\\begin{aligned}[t] a &= 1.25 \\\\[2pt] b &= 3 % note 99\n\\end{aligned}"
           r"\begin{array}{cc} 7 & 8 \end{array}\begin{alignat}{2} q &= 5 \end{alignat}"
           r"\kern-1.5pt\mskip 3mu\eqref{eq:ten}\label{eq:one}")
    assert nums(tex) == ["1.25", "3", "7", "8", "5"] and words(tex) == ["a", "b", "q"]
    assert nums(r"x \tag{4}") == ["4"]                          # a tag prints


def test_text_check_reads_typeset_math():
    """pdftotext gives mathematical italic letters, U+2212 and glued sub- and
    superscripts; the check folds them and still catches a lost number."""
    draft = ("The prior" + B.math_text(r"\mu_0 = -2.5") + "and"
             + B.math_text(r"\mathbb{E}[p(1-p)] = B_{31}") + "with"
             + B.math_text(r"\sigma_d^2 + \sigma_g^2 = 2.671^2 + 1.542^2") + "and"
             + B.math_text(r"\operatorname{logit} \hat p_i") + "here.")
    pdf = ("The prior \U0001d707\u2080 = \u22122.5 and \U0001d53c[\U0001d45d(1 \u2212 \U0001d45d)] = "
           "\U0001d43531 with \U0001d70e\U0001d4512 + \U0001d70e\U0001d4542 = 2.6712 + 1.5422 "
           "and logit \U0001d45d\u0302\U0001d456 here.")
    tc = B.text_check(draft, pdf)
    assert tc["numbers_missing"] == {} and tc["words_missing"] == {}
    lost = B.text_check(draft, pdf.replace("\u22122.5", "\u2212"))
    assert lost["numbers_missing"] == {"2.5": 1}


@needs_pandoc
def test_draft_plain_reads_math_as_printed():
    txt = B.draft_plain(PANDOC, "# T\n\nA $\\mu_0 = -2.5$ and $5 and $6.\n\n$$\n"
                                "\\mathrm{ALC} = 0.1\\,B_{31}\n$$\n\nGitHub's $`\\tau_{7}`$ "
                                "form.\n\n```math\n\\text{gap} = 0.0419 % 99\n```\n")
    assert "-2.5" in txt and "ALC" in txt and "31" in txt and "$5 and $6" in txt
    assert "7" in txt and "gap" in txt and "0.0419" in txt and "99" not in txt
    assert "mathrm" not in txt and "mu" not in txt and "tau" not in txt and "\\" not in txt


@needs_pandoc
def test_filter_leaves_math_intact(mini_math, tmp_path):
    r = B.build(mini_math, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    tex = Path(r["tex"]).read_text(encoding="utf-8").split("\\begin{document}", 1)[1]
    flt = r["filter"]
    assert flt["warnings"] == [] and flt["unresolved"] == []
    assert flt["links"]["table"] == 1 and flt["links"]["figure"] == 1
    assert flt["links"]["appendix"] == 1
    # inline and display math reach LaTeX as written: no minus sign, break point or link
    # inserted inside it
    assert "\\(\\mu_0 = -2.5\\)" in tex and "\\(\\Delta = -0.0419\\)" in tex
    assert "\\[\n\\mathrm{ALC} = 0.1\\,B_0" in tex
    assert "\\operatorname{logit}^{-1}(\\eta_{si})" in tex
    assert "\\(\\operatorname{cap}(o) = 4\\tanh(o/4)\\)" in tex     # in a table cell
    assert "\\(\\sqrt{1 - r^2}\\)" in tex                            # in a figure caption
    assert "−0.5" in tex                                             # text still gets its minus
    assert "\\$5 and \\$6" in tex                                    # a price stays text
    # in the abstract, emphasis, a list, a table caption and a table note
    assert "\\(B_{31} = 0.1725\\)" in tex
    assert "\\emph{the level \\(\\mu_b\\) in emphasis}" in tex
    assert "\\textbf{a gain of \\(-0.042\\) in bold}" in tex
    assert "\\(p ≤ 0.05\\)" in tex and "p ≤ 0.06 → text" in tex
    assert "\\emph{Table 1: the level prior \\(\\mu_b\\).}" in tex
    note = tex.index("\\(\\pm 0.01\\)")
    assert tex.rindex("\\begingroup\\small", 0, note) > tex.rindex("\\end{tabular}", 0, note)
    # a display stays a display; one amsmath environment is written as itself, starred
    assert "\\[\n\\begin{aligned}\na &= 1.25 \\\\[2pt]\n" in tex
    assert "\\begin{align*}\nx &= 7.5 \\\\\ny &\\le 8.5\n\\end{align*}" in tex
    assert "\\begin{align}" not in tex
    # a math cell is measured: the table keeps natural widths and one page
    assert flt["tables"][0]["layout"] == "natural" and flt["tables"][0]["kept_together"]


@needs_pandoc
@pytest.mark.parametrize("old, new, warning", [
    ("### A.1 One", "### A.1 The prior on $\\mu_b$", "holds TeX math"),
    ("Technical report for a test.", "Technical report on $\\mu_b$.", "the subtitle holds TeX math"),
    ("It costs $5", "A level $ \\mu_b$ costs $5", "left as text"),
    ("near 0.128.", "near $0.128 $.", "left as text"),
])
def test_filter_reports_math_it_cannot_print_well(mini_math, tmp_path, old, new, warning):
    """Math in a heading or the subtitle would show as TeX in the PDF's bookmarks and
    metadata; a "$" beside a letter that stayed text is math that did not parse."""
    assert old in MATH
    mini_math.write_text(MATH.replace(old, new), encoding="utf-8")
    r = B.build(mini_math, tmp_path / "out", figures="png", tex_only=True, epoch=0, quiet=True)
    assert [w for w in r["filter"]["warnings"] if warning in w], r["filter"]["warnings"]


@needs_latex
def test_math_pdf_text_check(mini_math, tmp_path):
    r = B.build(mini_math, tmp_path / "out", figures="png", epoch=1_700_000_000, quiet=True)
    log = r["log"]
    assert log["missing_glyphs"] == [] and log["pdf_string_tokens"] == 0
    assert [o for o in log["overfull"] if o["pt"] > B.OVERFULL_LIMIT_PT] == []
    tc = r["text_check"]
    if "skipped" not in tc:
        assert tc["numbers_missing"] == {} and tc["words_missing"] == {}
        assert tc["numbers"] >= 40


@needs_latex
def test_math_with_the_fallback_fonts(mini_math, tmp_path, monkeypatch):
    """Without Times New Roman, STIXGeneral and STIX Two Math (TeX Live alone) the text
    and the math fall back to TeX Gyre Termes, found by file name, and a "≤" typed in
    math is still set (as the math relation, by the symbol fallback)."""
    kit = tmp_path / "kit"
    shutil.copytree(B.KIT, kit)
    tpl = (kit / "template.tex").read_text(encoding="utf-8")
    for face in ("Times New Roman", "STIXGeneral", "STIX Two Math"):
        assert "\\IfFontExistsTF{" + face + "}" in tpl
        tpl = tpl.replace("\\IfFontExistsTF{" + face + "}", "\\IfFontExistsTF{No " + face + "}")
    (kit / "template.tex").write_text(tpl, encoding="utf-8")
    monkeypatch.setattr(B, "KIT", kit)
    r = B.build(mini_math, tmp_path / "out", figures="png", epoch=1_700_000_000, quiet=True)
    log = r["log"]
    assert log["missing_glyphs"] == [] and log["pdf_string_tokens"] == 0
    assert [o for o in log["overfull"] if o["pt"] > B.OVERFULL_LIMIT_PT] == []
    tc = r["text_check"]
    if "skipped" not in tc:
        assert tc["numbers_missing"] == {} and tc["words_missing"] == {}
    pypdf = pytest.importorskip("pypdf")
    fonts = set()
    for page in pypdf.PdfReader(r["pdf"]).pages:
        for f in page["/Resources"]["/Font"].values():
            # "ABCDEF+TeXGyreTermes-Regular-Identity-H": the subset prefix dropped
            fonts.add(str(f.get_object()["/BaseFont"]).split("+")[-1])
    for face in ("TeXGyreTermes-Regular", "TeXGyreTermes-Italic", "TeXGyreTermesMath-Regular"):
        assert any(f.startswith(face) for f in fonts), fonts
    assert not any(f.startswith(("TimesNewRoman", "STIX")) for f in fonts), fonts


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
    # --strict refuses only the placeholders and TODO markers left for the team (and a
    # main text over the limit, which test_report_main_text_within_the_limit checks)
    assert all(p.startswith(("title block placeholders", f"{len(r['todo_markers'])} TODO markers",
                             "main text is "))
               for p in B.problems(r))
    assert r["style"]["ok"] and r["inputs"]["tools/report_pdf/neurips_2026.sty"] == B.STYLE_SHA256
    if isinstance(r["fonts"], list):
        assert r["fonts"] and B.bad_fonts(r["fonts"]) == []
    tc = r["text_check"]
    if "skipped" not in tc:
        assert tc["numbers_missing"] == {} and tc["words_missing"] == {}


@needs_latex
def test_report_pdf_layout(built):
    r = built
    pages, lp = r["pages"], r["label_pages"]
    assert pages["main_text"][0] == 1 and pages["appendices"][1] == pages["total"]
    assert pages["main_text"][1] <= pages["references"][0] < pages["appendices"][0]
    assert pages["main_text_basis"].startswith("end mark")
    # §1 starts on the title page or the next one: report v2's title (three lines) and
    # abstract fill page 1, so §1 opens page 2 (v1.0.1 opened it on page 1)
    assert lp["sec-1"] <= 2 and lp["references"] < lp["report:appendix-start"]
    labels = B.figures_md_placement(FIGURES_MD.read_text(encoding="utf-8"))
    main = [lab for lab in labels if lab.isdigit()]
    for lab in labels:
        page = lp[f"fig-{lab}"]
        if lab in main:
            assert page <= min(lp["references"], pages["main_text"][1]), lab
        else:
            assert page >= lp["report:appendix-start"], lab
    assert [lp[f"fig-{lab}"] for lab in main] == sorted(lp[f"fig-{lab}"] for lab in main)


@needs_latex
def test_report_main_text_within_the_limit(built):
    """At most MAX_MAIN_PAGES (8) pages of main text in the NeurIPS template, references
    and appendices excluded, as the competition asks; every main-text heading is
    measured."""
    pages = built["pages"]
    sections = [s for s in built["sections"] if s["main"]]
    assert sections and all("pages" in s and s["words"] > 0 for s in sections)
    # it ends where the acknowledgments begin, which do not count (nor references and
    # appendices); the section's page is between the main text and References
    assert pages["main_text_ends_before"] == B.ACK_TITLE
    assert pages["main_text"][1] <= pages["acknowledgments"] <= pages["references"][0]
    assert not {s["id"]: s for s in built["sections"]}[ACK_ID]["main"]
    assert pages["main_text_pages"] <= B.MAX_MAIN_PAGES, (
        f"main text {pages['main_text_pages']} pages ({pages['main_text_measured']} to its last "
        f"line): " + "; ".join(f"{s['title'][:30]} {s['pages']}" for s in sections
                               if s["level"] == 2))


@needs_latex
def test_report_pdf_metadata_names_the_author_only(built):
    pdf = Path(built["pdf"]).read_bytes()
    assert b"%PDF-" == pdf[:5]
    pypdf = pytest.importorskip("pypdf")
    info = pypdf.PdfReader(built["pdf"]).metadata
    assert info.author == B.front_matter(DRAFT.read_text(encoding="utf-8"))["author"]
    assert info.title == B.front_matter(DRAFT.read_text(encoding="utf-8"))["title"]
    assert info.get("/Keywords") == B.front_matter(DRAFT.read_text(encoding="utf-8"))["keywords"]
    assert not any(EMAIL_RE.search(str(v)) for v in info.values())


@needs_latex
def test_acknowledgments_do_not_count_as_main_text(mini, tmp_path):
    """A small draft built without and with the acknowledgments section: the main text
    keeps its pages and is measured to the acknowledgments heading, not to References;
    build.json names the heading it ends before and the section's page, the PDF prints
    the style's heading and the sentence, and its outline lists it. (The measure itself
    may move a little: the style sets flush-bottom pages, so what else fits on the page
    changes how much its glue stretches.)"""
    plain = B.build(mini, tmp_path / "plain", figures="png", epoch=1_700_000_000, quiet=True)
    _with_ack(mini)
    ack = B.build(mini, tmp_path / "ack", figures="png", epoch=1_700_000_000, quiet=True)
    p0, p1 = plain["pages"], ack["pages"]
    assert p0["main_text_ends_before"] == "References" and "acknowledgments" not in p0
    assert p1["main_text_ends_before"] == B.ACK_TITLE
    assert p1["main_text"] == p0["main_text"] and p1["main_text_pages"] == p0["main_text_pages"]
    assert p1["acknowledgments"] == ack["label_pages"][B.ACK_LABEL] <= p1["references"][0]
    rows0 = {s["id"]: s for s in plain["sections"]}
    rows = {s["id"]: s for s in ack["sections"]}
    assert rows0["references"]["start"] == p0["main_text_measured"]
    assert rows[ACK_ID]["start"] == p1["main_text_measured"] < rows["references"]["start"]
    assert not rows[ACK_ID]["main"] and rows[ACK_ID]["pages"] > 0
    assert B.problems(ack) == B.problems(plain)
    assert ack["log"]["undefined_refs"] == [] and ack["log"]["missing_glyphs"] == []
    tc = ack["text_check"]
    if "skipped" not in tc:
        assert tc["numbers_missing"] == {} and tc["words_missing"] == {}
    pdftotext = B.find_tool("pdftotext", "PDFTOTEXT")
    if pdftotext:
        text = " ".join(B.pdf_plain(pdftotext, Path(ack["pdf"])).split())
        assert f"{B.ACK_TITLE} {ACK_SENTENCE}" in text
        assert text.index(ACK_SENTENCE) < text.index("Brier, G. W. (1950)")
    pypdf = pytest.importorskip("pypdf")

    def titles(outline):
        for x in outline:
            yield from (titles(x) if isinstance(x, list) else [x.title])

    assert B.ACK_TITLE in list(titles(pypdf.PdfReader(ack["pdf"]).outline))


@needs_latex
def test_build_is_deterministic(mini, tmp_path):
    a = B.build(mini, tmp_path / "a", figures="png", epoch=1_700_000_000, quiet=True)
    b = B.build(mini, tmp_path / "b", figures="png", epoch=1_700_000_000, quiet=True)
    assert a["pdf_sha256"] == b["pdf_sha256"]
    assert hashlib.sha256(Path(a["pdf"]).read_bytes()).hexdigest() == a["pdf_sha256"]
    assert a["pages"]["total"] >= 2 and a["pages"]["appendices"][0] == a["pages"]["total"]
    # the small draft's main text ends on its first page; References follows it
    assert a["pages"]["main_text"] == [1, 1] and a["pages"]["references"][0] >= 1
    assert 0 < a["pages"]["main_text_measured"] <= 1
    rows = {s["id"]: s for s in a["sections"]}
    assert rows["sec-1"]["main"] and 0 < rows["sec-1"]["pages"] < 1
    # the footer of the first page names the workshop
    pdftotext = B.find_tool("pdftotext", "PDFTOTEXT")
    if pdftotext:
        import subprocess
        first = subprocess.run([pdftotext, "-f", "1", "-l", "1", "-enc", "UTF-8", a["pdf"], "-"],
                               capture_output=True, text=True).stdout
        assert "Workshop: Predictive AI Evaluation Competition (PAIEC)" in " ".join(first.split())
