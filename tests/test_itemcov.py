"""paiec/itemcov.py: item covariates with a known sign, as pure functions of the
item dict (no data needed), and the sign rule and scaling of
experiments/itemcov_eval.py on synthetic numbers."""
import math

import numpy as np
import pytest

from paiec import itemcov as IC


def item(content="", features="", interactors="", bench="benchmark_123456"):
    return {"item_content": content, "item_features": features, "interactors": interactors,
            "benchmark_id": bench}


# --- never raise ---------------------------------------------------------------------------

BAD = [None, {}, [], "text", 3, {"item_content": None, "item_features": None},
       {"item_content": 5, "item_features": 7, "interactors": object()},
       {"item_content": float("nan"), "item_features": "difficulty=nan"},
       {"item_content": "x" * 10, "item_features": "{not json"},
       {"item_content": "\n" * 5000 + "A) 1\nB) 2\nC) 3", "item_features": "=;=;;=="}]


@pytest.mark.parametrize("bad", BAD)
def test_nothing_raises_and_every_value_is_finite_or_none(bad):
    out = IC.covariates(bad)
    assert set(out) == set(IC.COVARIATES)
    for v in out.values():
        assert v is None or (isinstance(v, float) and math.isfinite(v))
    assert IC.answer_format(bad) in IC.FORMAT_SCORE
    assert isinstance(IC.n_options(bad), int)


def test_a_plain_item_carries_no_ordinal_cue():
    out = IC.covariates(item("Fix the crash when the config file is empty.", "lang=go"))
    assert out["ordinal_difficulty"] is None and out["position"] is None and out["stated_size"] is None
    assert out["image_ref"] == 0.0 and out["format_score"] == 0.0
    assert out["log_length"] == pytest.approx(math.log1p(len("Fix the crash when the config file is empty.")))


# --- ordinal difficulty fields ----------------------------------------------------------------

@pytest.mark.parametrize("key,kind", [
    ("difficulty", "difficulty"), ("Difficulty_Level", "difficulty"), ("gradeLevel", "difficulty"),
    ("tier", "difficulty"), ("stars", "difficulty"), ("codeforces_rating", "difficulty"),
    ("n_steps", "work"), ("num_files", "work"), ("hops", "work"),
    ("problem_idx", "position"), ("question_number", "position"), ("taskIndex", "position"),
    ("repo_stars", None), ("log_level", None), ("user_rating", None), ("problem_id", None),
    ("competition", None), ("lang", None), ("website", None), ("paper", None), ("prompt_source", None),
])
def test_key_kinds(key, kind):
    assert IC.key_kind(key) == kind


@pytest.mark.parametrize("value,x", [
    ("3", 3.0), ("Level 3", 3.0), ("level_4", 4.0), ("L2", 2.0), ("3/5", 3.0), ("3 stars", 3.0),
    ("★★★", 3.0), ("tier-2", 2.0), ("-1", -1.0), ("easy", -1.0), ("Medium", 0.0), ("HARD", 1.0),
    ("very hard", 2.0), ("medium-hard", 0.5), ("hard problem", 1.0),
    ("aime_2025", None), ("", None), (None, None), ("x" * 100, None),
])
def test_ordinal_values(value, x):
    assert IC.ordinal_value(value) == x


def test_words_are_ordered_easy_to_hard():
    order = ["trivial", "easy", "easy-medium", "medium", "medium-hard", "hard", "expert"]
    vals = [IC.ordinal_value(w) for w in order]
    assert vals == sorted(vals) and len(set(vals)) == len(vals)


def test_difficulty_key_is_read_whatever_the_feature_format():
    for feats in ("difficulty=hard;topic=algebra", '{"difficulty": "hard", "topic": "algebra"}',
                  "topic: algebra\ndifficulty: hard"):
        assert IC.ordinal_difficulty(item("q", feats)) == 1.0
    assert IC.ordinal_difficulty(item("q", "level=5")) == 5.0
    # interactors are read too
    assert IC.ordinal_difficulty(item("q", "", "tier=2")) == 2.0
    # a work count, when no difficulty key: log(1 + count)
    assert IC.ordinal_difficulty(item("q", "n_steps=7")) == pytest.approx(math.log1p(7))
    # a difficulty key wins over a work count
    assert IC.ordinal_difficulty(item("q", "n_steps=7;difficulty=easy")) == -1.0
    # a key that only looks like one is ignored
    assert IC.ordinal_difficulty(item("q", "repo_stars=1200;log_level=3")) is None


def test_matharena_problem_idx_is_a_position():
    it = item("Please reason step by step, and put your final answer within \\boxed{}.",
              "competition=aime_2025;problem_idx=13;prompt_source=user_message")
    assert IC.position(it) == pytest.approx(math.log1p(13))
    assert IC.ordinal_difficulty(it) is None
    assert IC.position(item("q", "problem_idx=abc")) is None


# --- answer format ---------------------------------------------------------------------------

def test_formats():
    proof = item("Your task is to write a proof solution to the following problem.\n\nLet n be ...")
    imo = item("Let $ABC$ be a triangle.\n\nProve that the line through $H$ is tangent.")
    kangaroo = item("Reason step by step, referring to the given multiple choice options (A, B, C, D, "
                    "or E), of which exactly one is correct.\n\nSee image.")
    listed = item("Which is prime?\nA) 4\nB) 6\nC) 7\nD) 9")
    short = item("Please reason step by step, and put your final answer within \\boxed{}.\nFind x.")
    code = item("Crash on empty input\n```python\nparse('')\n```")
    free = item("Show me places in Delhi for 2 guests.")
    assert [IC.answer_format(x) for x in (proof, imo, kangaroo, listed, short, code, free)] == \
        ["proof", "proof", "mcq", "mcq", "short", "code", "free"]
    assert IC.n_options(kangaroo) == 5 and IC.n_options(listed) == 4
    assert IC.format_score(kangaroo) < IC.format_score(short) < IC.format_score(proof)
    # a paper's "we prove that" is not an instruction to prove
    assert IC.answer_format(item("Abstract. In this paper we prove that SGD converges.\n```python\nx = 1\n```")) \
        == "code"


def test_drawing_labels_are_not_answer_options():
    tikz = ("Find the area.\n\\begin{tikzpicture}\n\\coordinate (A) at (0,0);\n\\coordinate (B) at (1,0);\n"
            "\\coordinate (C) at (1,1);\n\\coordinate (D) at (0,1);\n\\coordinate (E) at (2,2);\n"
            "\\draw (A) -- (B);\n\\end{tikzpicture}\nPut your final answer within \\boxed{}.")
    assert IC.n_options(item(tikz)) == 0
    assert IC.answer_format(item(tikz)) == "short"


def test_asymptote_and_prose_are_not_code():
    asy = item("The graph of $f$ is shown.\n\\begin{asy}\nimport graph;\nsize(8cm);\n\\end{asy}\nCompute f(3).")
    prose = item("Let\n$$f(x) = x^2$$\ndefined over the reals. Compute f(3).")
    assert IC.answer_format(asy) == "short" and IC.answer_format(prose) == "short"


# --- stated size and image references -----------------------------------------------------------

def test_stated_size_takes_the_first_statement():
    rcb = item("class A:\n    def f(self):\n        # TODO: Implement block \"loss\"\n"
               "        # Approximately 7 line(s) of code.\n        pass\n\nFor example, if you see this nested "
               "TODO block:\n# TODO: Implement block \"calculate area\"\n# Approximately 2 line(s) of code.")
    assert IC.stated_size(rcb) == pytest.approx(math.log1p(7))
    assert IC.stated_size(item("Book a room in about 5 steps.")) == pytest.approx(math.log1p(5))
    assert IC.stated_size(item("It has 5 steps.")) is None          # no amount word: a description
    assert IC.stated_size(item("Solve it.")) is None


def test_image_references():
    for t in ("See image.", "see the figure below", "![screenshot](https://x/y.png)", "<image>", "<img src=a>"):
        assert IC.image_ref(item(t)) == 1.0
    assert IC.image_ref(item("The image module crashes")) == 0.0


def test_covariates_are_a_pure_function_of_the_item():
    it = item("Prove that 2 is prime.", "difficulty=hard;problem_idx=3")
    a, b = IC.covariates(it), IC.covariates(dict(it))
    assert a == b
    assert IC.covariates(dict(it, benchmark_id="benchmark_999999")) == a    # the id is never read


# --- the evaluation's pure helpers --------------------------------------------------------------

@pytest.fixture(scope="module")
def E():
    from experiments import itemcov_eval
    return itemcov_eval


def test_sign_rule_needs_four_units(E):
    def per(rhos):
        return {u: {"applies": r is not None, "rho": r} for u, r in zip(E.UNITS, rhos)}
    ok = E.sign_rule(per([0.3, 0.2, 0.1, 0.25, -0.05]), 1)
    assert ok["declared_sign_units"] == 4 and ok["lobo_agree_units"] == 4 and ok["transferred_allowed"]
    three = E.sign_rule(per([0.3, 0.2, 0.1, None, None]), 1)
    assert three["n_applicable"] == 3 and not three["transferred_allowed"]
    mixed = E.sign_rule(per([0.28, -0.02, 0.36, -0.10, 0.04]), 1)     # log_length's pattern
    assert mixed["declared_sign_units"] == 3 and not mixed["transferred_allowed"]
    wrong = E.sign_rule(per([-0.3, -0.2, -0.1, -0.25, -0.05]), 1)    # agree with each other, not the declared sign
    assert wrong["lobo_agree_units"] == 5 and not wrong["transferred_allowed"]
    # mmdocrag is directional only
    extra = dict(per([0.3, 0.2, 0.1, None, None]), mmdocrag={"applies": True, "rho": 0.4})
    assert E.sign_rule(extra, 1)["n_applicable"] == 3


def test_standardised_within_benchmark(E):
    rng = np.random.default_rng(0)
    xmap = {f"a{i}": float(v) for i, v in enumerate(rng.normal(5, 2, 200))}
    xmap.update({f"b{i}": 1.0 if i < 3 else 0.0 for i in range(100)})       # 3 off the mode: dropped
    xmap["a0"] = 1e6                                                         # clipped
    z = E.standardised(xmap, {"A": [f"a{i}" for i in range(200)], "B": [f"b{i}" for i in range(100)]})
    za = np.array([z[f"a{i}"] for i in range(200)])
    assert not any(k.startswith("b") for k in z)
    assert np.max(np.abs(za)) <= E.CLIP and za[0] == E.CLIP
    assert abs(np.median(za)) < 0.3


def test_rank_and_unit_corr(E):
    assert list(E._rank([3.0, 1.0, 2.0, 2.0])) == [4.0, 1.0, 2.5, 2.5]
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(10), 20)
    x = rng.normal(size=200)
    d = 3 * g + x + 0.1 * rng.normal(size=200)          # groups differ, x orders items within them
    out = E.unit_corr(x, d, g.astype(str), boot=50)
    assert out["rho_within"] > 0.9 and out["rho"] < out["rho_within"]
    assert out["boot_unit"] == "group" and out["rho_within_se"] >= 0
