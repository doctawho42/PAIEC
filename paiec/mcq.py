"""Guessing floor read straight off the item text.

A four-way multiple-choice item cannot be answered correctly less than a quarter
of the time by a system that always answers, so the predicted probability should
never drop below 1/n. This costs nothing: no labels, no training, just the text.
"""
import re

PATS = [re.compile(r"(?m)^\s*\(?([A-E])[\)\.]\s+"),      # A) / (A) / A. at line start
        re.compile(r"\(([A-E])\)\s"),                     # inline (A)
        re.compile(r"(?m)^\s*([A-E])\s*[:：]\s")]


def n_options(text):
    """Number of answer options, or 0 when the item is not multiple choice."""
    if not text:
        return 0
    best = set()
    for p in PATS:
        letters = {m.group(1).upper() for m in p.finditer(text)}
        if len(letters) > len(best):
            best = letters
    if len(best) < 3:
        return 0
    for k in range(len(best), 2, -1):
        if set("ABCDE"[:k]) <= best:
            return k
    return 0


def floor_of(text, cap=0.34):
    n = n_options(text)
    return min(1.0 / n, cap) if n else 0.0
