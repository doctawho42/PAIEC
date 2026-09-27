"""paiec/mcq.py: the multiple-choice guessing floor the shipped hier reads.

Two defects of the floor as it stood at commit bd0be67 (experiments/mcq_floor.py
holds that version as old_n_options): items whose options are only named in the
instructions, "(A, B, C, D, or E)" with the options in an image (the 336
matharena Kangaroo items), read as no options; labelled points of a TikZ drawing,
"\\coordinate (A) at (0,0);", read as options (A)..(E) on 20 integer-answer
matharena items. Everything else must read as before, and no text may cost more
than it did before by more than a linear pass. The public-item tests need data/
and are skipped without it.

The fix as replayed (experiments/mcq_floor.py's new_n_options) was revised
after review without changing what any public text reads: only closed drawing
blocks are cut (an unclosed one ran to the end of the text), a cut leaves a
non-blank placeholder, the list needs 'or' ('and' named points too) and its
whitespace has one reading (two adjacent '\\s*' made it quadratic).

"""
import ast
import inspect
import time

import pytest

from paiec import mcq

from experiments import mcq_floor as MF
from paiec.hier import mcq_text

KANGAROO = ("You are given a Math Kangaroo problem in the form of an image. Look at the statement "
            "carefully.\n Some problems may contain illustrations that are essential for solving them. "
            "Reason step by step, and put your final answer within \\boxed{}, referring to the given "
            "multiple choice options (A, B, C, D, or E), of which exactly one is correct.\nAn example of "
            "a valid answer is \\boxed{A}.\n\n\nSee image.")
TIKZ = ("Find the area of the shaded region.\n\\begin{center}\n\\begin{tikzpicture}[scale=0.5]\n"
        "\\coordinate (A) at (0,0);\n\\coordinate (B) at (6,0);\n\\coordinate (C) at (6,4);\n"
        "\\coordinate (D) at (0,4);\n\\coordinate (E) at (3,6);\n"
        "\\draw (A) -- (B) -- (C) -- (D) -- cycle;\n\\node at (A) [below left] {$A$};\n"
        "\\end{tikzpicture}\n\\end{center}\nThe answer is an integer between 0 and 999.")
OPTIONS = "\n(A) 12\n(B) 18\n(C) 24\n(D) 30"

#: ordinary multiple-choice texts in every form the patterns read, and texts
#: that are not multiple choice; each must read as it did before
ORDINARY = [
    ("Which is prime?\n(A) 4\n(B) 6\n(C) 7\n(D) 9", 4),
    ("Pick one: (A) 1 (B) 2 (C) 3 (D) 4 (E) 5", 5),
    ("Which is largest?\nA. 1\nB. 2\nC. 3\nD. 4", 4),
    ("Which is largest?\nA) 1\nB) 2\nC) 3", 3),
    ("Q\nA: red\nB: green\nC: blue\nD: black", 4),
    ("Q\nA： red\nB： green\nC： blue", 3),                    # full-width colon
    ("Q\nA：red\nB：green\nC：blue", 0),                       # no space after it
    ("Q\n\n   (A) x\n\n  (B) y\n (C) z\n", 3),
    ("x\r\nA. a\r\nB. b\r\nC. c\r\n", 3),
    ("A)\nB)\nC)", 0),                                          # no space after the letter
    ("Two options only: (A) yes (B) no", 0),
    ("Letters not from A: (B) 1 (C) 2 (D) 3", 0),
    ("Let $T=\\operatorname{conv}(A,B,C)$ be a lattice triangle. Find its area.", 0),
    ("Crash on empty input\n```python\nparse('')\n```", 0),
    ("Find the remainder when 2^100 is divided by 7.", 0),
]


def test_a_kangaroo_item_has_five_options():
    assert MF.old_n_options(KANGAROO) == 0                      # the defect
    assert mcq.n_options(KANGAROO) == 5
    assert mcq.floor_of(KANGAROO) == pytest.approx(0.2)
    assert mcq.floor_of(mcq_text(KANGAROO)) == pytest.approx(0.2)     # what hier reads


@pytest.mark.parametrize("text, n", [
    ("Answer (A, B, or C).", 3),
    ("Choose among the options (A, B, C, or D) shown in the figure.", 4),
    ("options (A,B,C,D,or E)", 5),
    ("options ( A , B , C , D or E )", 5),
    ("options (A,\n B,\n C,\n D, or\n E)", 5),
    ("the options (A, B, C, D or E) in the image", 5),
    ("Answer (A, B or C).", 3),
    ("options (A or B)", 0),
    ("options (A, C, or D)", 0),                                # not A, B, C, ... in order
    ("options (B, C, D, or E)", 0),
    ("options (A, B)", 0),
    ("points (A, B, C) of the triangle", 0),                    # no 'or'
    ("options (A, B, C, D, E, or F)", 6),
    ("options (A, B, C, D, E, F, G, or H)", 8),
    ("options (A, B, C, orD)", 0),                              # 'or' needs a space after it
    ("options (a, b, c, or d)", 0),                             # capitals only
])
def test_a_listed_set_of_options(text, n):
    assert mcq.n_options(text) == n


@pytest.mark.parametrize("text", [
    "Points (A, B, and C) are the vertices of an equilateral triangle of side 2. Find its area.",
    "The square has vertices (A, B, C, and D) in counterclockwise order; find the length of AC.",
    "options ( A , B , C , D and E )",
])
def test_a_list_joined_by_and_names_points_not_options(text):
    assert MF.new_n_options(text) >= 3                          # the replayed version read these
    assert mcq.n_options(text) == 0
    assert mcq.floor_of(mcq_text(text)) == 0.0


@pytest.mark.parametrize("text", [
    "def area(A, B, C):\n    return (A, B, C)\n",
    "x = (A, B, C)\ny = (A, B, C, D)\nprint(x, y)",
    "assert f(A, B, C) == (A, B, C)",
])
def test_a_code_tuple_is_not_a_list_of_options(text):
    assert mcq.n_options(text) == 0 and MF.old_n_options(text) == 0


def test_drawing_labels_are_not_options():
    assert MF.old_n_options(TIKZ) == 5                          # the defect
    assert mcq.n_options(TIKZ) == 0
    assert mcq.floor_of(TIKZ) == 0.0
    assert mcq.floor_of(mcq_text(TIKZ)) == 0.0
    three = "Find x.\n\\begin{tikzpicture}\n\\coordinate (A) at (0,0);\n\\coordinate (B) at (1,0);\n" \
            "\\coordinate (C) at (0,1);\n\\end{tikzpicture}\nGive an integer."
    assert MF.old_n_options(three) == 3 and mcq.n_options(three) == 0


@pytest.mark.parametrize("text", [
    "The graph is shown.\n\\begin{asy}\nlabel(\"(A) \", (0,0));\nlabel(\"(B) \", (1,0));\n"
    "label(\"(C) \", (2,0));\n\\end{asy}\nCompute f(3).",
    "[asy]\nlabel(\"(A) \", (0,0)); label(\"(B) \", (1,0)); label(\"(C) \", (2,0));\n[/asy]\nFind the sum.",
    "\\begin{picture}(10,10)\n\\put(0,0){(A) }\\put(1,0){(B) }\\put(2,0){(C) }\n\\end{picture}\nFind n.",
    "Drawn with \\begin{tikzpicture} below.\n[asy]\nlabel(\"(A) \"); label(\"(B) \"); label(\"(C) \");\n"
    "[/asy]\nFind x.",                                          # a closed block after an unclosed tag
])
def test_other_drawings_are_cut_too(text):
    assert MF.old_n_options(text) >= 3 and mcq.n_options(text) == 0


def test_explicit_options_outrank_a_listed_set():
    assert mcq.n_options("Answer (A, B, C, D, or E)\n(A) 1\n(B) 2\n(C) 3") == 3


def test_options_outside_a_drawing_still_count():
    text = TIKZ.replace("The answer is an integer between 0 and 999.", "What is the area?" + OPTIONS)
    assert mcq.n_options(text) == 4


@pytest.mark.parametrize("text", [
    "The figure, drawn with \\begin{tikzpicture} in the source, is omitted." + OPTIONS,
    "AoPS drawings start with [asy]; this one is omitted." + OPTIONS,
    "Which environment draws pictures? \\begin{picture} or \\begin{asy}?" + OPTIONS,
])
def test_options_after_an_unclosed_tag_still_count(text):
    """An opening tag without its closing tag is prose, not a drawing: the
    replayed version cut from it to the end of the text, options and all."""
    assert MF.new_n_options(text) == 0                          # the replayed version's defect
    assert mcq.n_options(text) == 4 == MF.old_n_options(text)
    assert mcq.floor_of(mcq_text(text)) == pytest.approx(0.25)


def test_options_after_a_drawing_whose_end_the_excerpt_dropped():
    """hier reads the first and last 10k characters; a long drawing loses its
    \\end, 37 researchcodebench excerpts do (they read no options either way)."""
    drawing = "\\begin{tikzpicture}\n" + "\\draw (0,0) -- (1,1);\n" * 700 + "\\end{tikzpicture}\n"
    text = "Look at the figure.\n" + drawing + "Some prose here.\n" * 1000 + "Which is right?" + OPTIONS
    view = mcq_text(text)
    assert "\\begin{tikzpicture}" in view and "\\end{tikzpicture}" not in view
    assert MF.new_n_options(view) == 0                          # the replayed version's defect
    assert mcq.n_options(view) == 4 and mcq.n_options(text) == 4


def test_an_unclosed_drawing_reads_as_before():
    """Nothing is cut, so its labelled points read as they did at bd0be67."""
    text = "Unclosed drawing.\n\\begin{tikzpicture}\n\\coordinate (A) at (0,0);\n" \
           "\\coordinate (B) at (1,0);\n\\coordinate (C) at (1,1);\n"
    assert mcq.n_options(text) == MF.old_n_options(text) == 3


def test_a_cut_drawing_leaves_no_blank_line():
    assert mcq.cut_drawings("a\n[asy]x[/asy]\nb") == "a\n _ \nb"
    assert mcq.cut_drawings("\\begin{asy}x\\end{tikzpicture}y\\end{asy}z") == " _ z"   # its own kind
    assert mcq.cut_drawings("[asy] only in prose") == "[asy] only in prose"
    cut = mcq.cut_drawings(mcq_text("[asy][/asy]\n" * 50 + "\\begin{tikzpicture}\\end{tikzpicture}\n" * 50))
    assert cut.count(" _ ") == 100 and all(line.strip() for line in cut.splitlines())


@pytest.mark.parametrize("text, n", ORDINARY)
def test_ordinary_texts_read_as_before(text, n):
    assert MF.old_n_options(text) == n
    assert mcq.n_options(text) == n
    assert mcq.floor_of(text) == MF.old_floor_of(text)
    assert mcq.floor_of(mcq_text(text)) == MF.old_floor_of(mcq_text(text))


@pytest.mark.parametrize("text", [None, "", "   ", "\n\n"])
def test_empty_text_has_no_floor(text):
    assert mcq.n_options(text) == 0
    assert mcq.floor_of(text) == 0.0


def test_the_floor_is_one_over_n_capped():
    assert mcq.floor_of("Q\n(A) 1\n(B) 2\n(C) 3") == pytest.approx(1 / 3)
    assert mcq.floor_of("Q\n(A) 1\n(B) 2\n(C) 3", cap=0.3) == pytest.approx(0.3)
    assert mcq.floor_of("Q\n(A) 1\n(B) 2\n(C) 3\n(D) 4") == pytest.approx(0.25)


def test_stays_linear_on_pathological_texts():
    for bad in ("\\begin{tikzpicture}" * 20_000, "[asy]" * 50_000, "(A, " * 50_000,
                "(" * 200_000, "(A, B, " * 30_000 + "or"):
        t0 = time.perf_counter()
        mcq.n_options(bad)
        mcq.floor_of(mcq_text(bad))
        assert time.perf_counter() - t0 < 0.5


def _secs(fn, text, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(text)
        best = min(best, time.perf_counter() - t0)
    return best


#: texts on which a regex can go superlinear, as (name, text, views): 'raw' is
#: the text as given, 'mcq_text' what hier reads (at most 20k characters, blank
#: lines dropped); the replayed version took 3 s on the first (20k spaces) and
#: 0.2 s on '[asy][/asy]' lines, the shipped floor 0.3 ms and 0.5 ms
WORST = [
    ("(A, B + 20k spaces", "(A, B" + " " * 20_000 + "x", ("raw",)),
    ("(A, B + 19k spaces", "(A, B" + " " * 19_000 + "x", ("raw", "mcq_text")),   # kept by mcq_text
    ("(A, B, C + 19k tabs + or", "(A, B, C" + "\t" * 19_000 + "or", ("raw", "mcq_text")),
    ("[asy][/asy] lines x2000", "[asy][/asy]\n" * 2000, ("raw", "mcq_text")),
    ("tikz lines x2000", "\\begin{tikzpicture}\\end{tikzpicture}\n" * 2000, ("raw", "mcq_text")),
    ("20k blank lines", "\n" * 20_000 + "(A) 1\n(B) 2\n(C) 3", ("mcq_text",)),
    ("20k blank lines, drawings", ("\n" * 10 + "[asy]x[/asy]") * 2000, ("mcq_text",)),
    ("blank runs around a drawing", "\n" * 1000 + "[asy]x[/asy]" + "\n" * 1000, ("raw",)),
    ("150k code", ("def f(A, B, C):\n    return (A, B, C)  # or D\n" * 3500)[:150_000], ("raw", "mcq_text")),
    ("150k spaces", "(A, B" + " " * 150_000 + "x", ("raw",)),
    ("150k unclosed tags", "\\begin{tikzpicture}\\begin{asy}\\begin{picture}[asy]" * 3000, ("raw",)),
    ("150k closed drawings", "[asy]label(\"(A) \");[/asy]\n" * 5800, ("raw",)),
    ("150k lists", "(A, B, C" + ", A" * 50_000 + " x", ("raw",)),
    ("150k (", "(" * 150_000, ("raw",)),
    ("150k Kangaroo", "pad " * 37_000 + KANGAROO, ("raw", "mcq_text")),
]


@pytest.mark.parametrize("name, text, views", WORST, ids=[w[0] for w in WORST])
def test_no_slower_than_the_shipped_floor(name, text, views):
    """The fix adds linear passes to the shipped patterns, nothing more: on the
    same text it costs at most 10x the shipped floor plus 50 ms (it measured
    at most 6x, 12 ms on 150k characters); a quadratic pass costs seconds."""
    for view in views:
        x = text if view == "raw" else mcq_text(text)
        old, new = _secs(MF.old_floor_of, x), _secs(mcq.floor_of, x)
        assert new <= 10 * old + 0.05, (view, len(x), old, new)


def test_the_module_imports_only_the_standard_library():
    """It ships in the submission archive as paiec_rt.mcq."""
    tree = ast.parse(inspect.getsource(mcq))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"re"}


def test_it_is_the_floor_experiments_mcq_floor_measured():
    for text, _ in ORDINARY:
        assert mcq.n_options(text) == MF.new_n_options(text)
    for text in (KANGAROO, TIKZ, "", None) + MF.UNIT_TEXTS:
        assert mcq.n_options(text) == MF.new_n_options(text)


#: every public item whose option count changes, by benchmark, group
#: (matharena competition), before, after and kind: the two known defects, nothing else
EXPECTED = {("matharena", "aime_2025", 5, 0, "drawing"): 12,
            ("matharena", "aime_2025_I", 5, 0, "drawing"): 3,
            ("matharena", "aime_2025_II", 5, 0, "drawing"): 3,
            ("matharena", "hmmt_feb_2025", 3, 0, "drawing"): 2,
            **{("matharena", f"kangaroo_2025_{g}", 0, 5, "listed"): n
               for g, n in (("1_2", 48), ("3_4", 48), ("5_6", 60), ("7_8", 60), ("9_10", 60),
                            ("11_12", 60))}}
NEEDS_DATA = pytest.mark.skipif(MF._read_items("matharena") is None, reason="needs data/ (python -m paiec.fetch)")


@NEEDS_DATA
def test_public_items_change_only_where_the_defects_are():
    table = MF.item_table(before=MF.old_n_options, after=mcq.n_options)
    assert set(table) == set(MF.BENCHMARKS)
    for view in ("mcq_text", "raw"):
        got = {(b, c["group"], c["before"], c["after"], c["kind"]): c["items"]
               for b, rec in table.items() for c in rec[view]["changes"]}
        assert got == EXPECTED, view
    assert table["matharena"]["mcq_text"]["floored_after"] == 336
    assert sum(EXPECTED.values()) == 356


@NEEDS_DATA
def test_public_texts_read_as_the_replayed_version_read_them():
    """The revision changes no public text's floor, raw or as hier reads it, so
    the replay (experiments/mcq_floor.py, results/mcq_floor.json) stands."""
    for text in MF.public_texts():
        for x in (text, mcq_text(text)):
            assert mcq.floor_of(x) == MF.new_floor_of(x)
