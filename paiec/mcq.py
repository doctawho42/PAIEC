"""Guessing floor read straight off the item text.

A four-way multiple-choice item cannot be answered correctly less than a quarter
of the time by a system that always answers, so the predicted probability should
never drop below 1/n. This costs nothing: no labels, no training, just the text.

Two things are read that the option patterns alone got wrong on the public
items (experiments/mcq_floor.py): drawing code is cut out first, because its
labelled points ("\\coordinate (A) at (0,0);" in TikZ, "(A)" in Asymptote)
read as options (A)..(E) on 20 integer-answer matharena items; and an item
whose options are only named in its instructions, "(A, B, C, D, or E)" with
the options themselves in an image (the 336 Kangaroo items), gets that many.
Neither costs more than a linear pass: the worst case stays that of the option
patterns alone (a run of blank lines, which hier.mcq_text removes).
Known limits, none of which touches a public item: an 'or' list of letters
A, B, C, ... in order also reads as options where it is code or prose (a call
f(A, B, or C), a list of grades); and an opening drawing tag with no drawing of
its own pairs with the next closing tag of its kind, so options between the two
are cut with it.
Standard library only: this module ships in the submission archive.
"""
import re

PATS = [re.compile(r"(?m)^\s*\(?([A-E])[\)\.]\s+"),      # A) / (A) / A. at line start
        re.compile(r"\(([A-E])\)\s"),                     # inline (A)
        re.compile(r"(?m)^\s*([A-E])\s*[:：]\s")]

#: where drawing code opens: TikZ, Asymptote (LaTeX environment or AoPS [asy]
#: tags) and LaTeX picture blocks
OPEN = re.compile(r"\\begin\{(tikzpicture|asy|picture)\}|\[asy\]")
#: what a cut drawing leaves: not whitespace, so a line of drawings does not
#: become a blank line for the '^\s*' patterns to rescan
CUT = " _ "
#: options named in the instructions: "(A, B, C, D, or E)", "(A, B or C)";
#: every whitespace run has one reading, so a failed match costs its length
LISTED = re.compile(r"\(\s*((?:[A-H]\s*,\s*)+[A-H]\s*(?:,\s*)?or\s+[A-H])\s*\)")


def cut_drawings(text):
    """text with every closed drawing block, from its opening tag to the first
    closing tag of its kind, replaced by CUT. An opening tag without one (a
    tag named in prose, or a drawing whose end the excerpt dropped) cuts
    nothing. One pass: a closing tag missing after one opening tag is missing
    after every later one, so it is looked for once."""
    out, at, pos, missing = [], 0, 0, set()
    while True:
        m = OPEN.search(text, pos)
        if m is None:
            break
        close = "\\end{%s}" % m.group(1) if m.group(1) else "[/asy]"
        end = -1 if close in missing else text.find(close, m.end())
        if end < 0:
            missing.add(close)
            pos = m.end()
            continue
        out += (text[at:m.start()], CUT)
        at = pos = end + len(close)
    out.append(text[at:])
    return "".join(out)


def _listed(text):
    """Options named in a parenthesised list, A first and in order, at least
    three; 0 when there is none."""
    for m in LISTED.finditer(text):
        letters = re.findall(r"[A-H]", m.group(1))
        k = len(letters)
        if k >= 3 and "".join(letters) == "ABCDEFGH"[:k]:
            return k
    return 0


def n_options(text):
    """Number of answer options, or 0 when the item is not multiple choice."""
    if not text:
        return 0
    text = cut_drawings(text)
    best = set()
    for p in PATS:
        letters = {m.group(1).upper() for m in p.finditer(text)}
        if len(letters) > len(best):
            best = letters
    if len(best) >= 3:
        for k in range(len(best), 2, -1):
            if set("ABCDE"[:k]) <= best:
                return k
    return _listed(text)


def floor_of(text, cap=0.34):
    n = n_options(text)
    return min(1.0 / n, cap) if n else 0.0
