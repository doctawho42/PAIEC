"""experiments/formative_feedback.py: the formative feedback record and the
pieces of the per-pair reading that do not need data/.

The feedback tables in results/formative/ must reproduce the platform's scores
with the official weights, pair-averaged; the parser must refuse tables whose
sections disagree; the stored preregistration must be the script's own; and
the small estimators (matching, mixture, variance split, reweighting) must do
what their docstrings say on synthetic inputs.
"""
import json
import math
import os
import subprocess
import zipfile

import numpy as np
import pytest

from experiments import formative_feedback as FF

ROOT = FF.ROOT


def _read(name):
    with open(FF.FEEDBACK[name]) as f:
        return f.read()


@pytest.mark.parametrize("name, reported, n_pairs", [("run1", 0.211300, 9), ("run2", 0.192623, 8)])
def test_feedback_reproduces_the_platform_score(name, reported, n_pairs):
    pairs = FF.parse(_read(name))
    assert len(pairs) == n_pairs
    alc = FF.run_alc(pairs)
    assert round(alc, 6) == reported
    # per pair, the budgets give the summary row's ALC up to the table's rounding
    for p in pairs:
        assert abs(float(np.dot(FF.W, p["brier"])) - p["alc"]) < 1e-6
        assert abs(float(np.dot(FF.W, p["ece_b"])) - p["ece"]) < 1e-6
    # weighting pairs by their item count is not the platform's rule
    item_w = np.dot([p["alc"] for p in pairs], [p["n"] for p in pairs]) / sum(p["n"] for p in pairs)
    assert round(float(item_w), 6) != reported


def test_summary_counts_and_overlap():
    r1, r2 = FF.parse(_read("run1")), FF.parse(_read("run2"))
    s1 = FF.summarise_run("run1", r1)
    assert s1["counts"]["pairs"] == 9 and s1["counts"]["benchmarks"] == 7
    assert s1["alc"]["reproduces_reported"]
    ov = FF.overlap(r1, r2)
    assert ov["subjects"]["shared"] == [] and ov["pairs"]["shared"] == []
    assert ov["paired_comparison"] is None


def _table(rows_by_section):
    out = ["Formative feedback", ""]
    for head, rows in rows_by_section:
        out += [head, "", "subject_id  Benchmark  N  Brier  ECE", "---"]
        out += [f"{s}  {b}  {n}  {x:.6f}  {e:.6f}" for s, b, n, x, e in rows]
        out.append("")
    return "\n".join(out)


def _sections(rows):
    return [("ALC summary", rows)] + [(f"Label budget: {B}", rows) for B in FF.BUDGETS]


def test_parse_refuses_disagreeing_sections():
    rows = [("subject_1", "benchmark_1", 50, 0.2, 0.1), ("subject_2", "benchmark_1", 60, 0.1, 0.1)]
    pairs = FF.parse(_table(_sections(rows)))
    assert [p["n"] for p in pairs] == [50, 60] and pairs[0]["brier"] == [0.2] * 6
    swapped = _sections(rows)
    swapped[3] = (swapped[3][0], rows[::-1])
    with pytest.raises(ValueError):
        FF.parse(_table(swapped))
    with pytest.raises(ValueError):
        FF.parse(_table(_sections(rows)[:-1]))


def test_overlap_pairs_a_recurring_pair():
    a = FF.parse(_table(_sections([("subject_1", "benchmark_1", 50, 0.2, 0.1)])))
    b = FF.parse(_table(_sections([("subject_1", "benchmark_1", 52, 0.1, 0.1),
                                   ("subject_2", "benchmark_2", 44, 0.3, 0.1)])))
    ov = FF.overlap(a, b)
    assert ov["pairs"]["shared"] == [["subject_1", "benchmark_1"]]
    (row,) = ov["paired_comparison"]
    assert row["n"] == [50, 52] and math.isclose(row["run2_minus_run1_alc"], -0.1, abs_tol=1e-12)


def test_stored_preregistration_is_the_scripts():
    if not os.path.exists(FF.OUT):
        pytest.skip("no results/formative_feedback.json")
    with open(FF.OUT) as f:
        state = json.load(f)
    if "preregistration" not in state:
        pytest.skip("no preregistration stored")
    pre = state["preregistration"]
    body = {k: v for k, v in pre.items() if k not in ("sha256", "written_utc")}
    assert body == FF.PREREGISTRATION
    assert pre["sha256"] == FF.prereg_digest()
    # it was written before any reading
    when = {p["stage"]: p["when_utc"] for p in reversed(state["passes"])}
    if "read" in when:
        first_read = min(p["when_utc"] for p in state["passes"] if p["stage"] == "read")
        assert pre["written_utc"] < first_read


def test_neighbours_standardise_each_column():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 100, 200)])
    v = X[17] + np.array([0.0, 0.0])
    nn, d = FF.neighbours(X, v, 5)
    assert nn[0] == 17 and d[17] == 0
    # a unit step in the wide column is small after standardising
    w = X[17] + np.array([0.0, 50.0])
    nn2, _ = FF.neighbours(X, w, 1)
    assert abs(X[nn2[0], 0] - X[17, 0]) < 0.5


def test_mixture_prefers_two_components_only_when_there_are_two():
    two = np.r_[np.linspace(-4.2, -3.8, 8), np.linspace(1.8, 2.2, 9)]
    one = np.linspace(-1.5, 1.5, 17)
    assert FF.mixture(two)["delta_bic_one_minus_two"] > 2
    assert FF.mixture(one)["delta_bic_one_minus_two"] < 2
    m = FF.mixture(two)["two"]
    assert m["means"][0] < -3 and m["means"][1] > 1.5


def test_anova_split():
    # balanced: 4 groups of 3, group means -1, 0, 1, 2 exactly, within spread 0.5
    x, g = [], []
    for k, mu in enumerate((-1.0, 0.0, 1.0, 2.0)):
        for e in (-0.5, 0.0, 0.5):
            x.append(mu + e)
            g.append(k)
    a = FF.anova(x, g)
    msw = 0.25 * 2 / 2      # per group: (0.25 + 0 + 0.25) / (3 - 1)
    msb = 3 * np.var([-1, 0, 1, 2], ddof=1)
    assert math.isclose(a["within_sd"], math.sqrt(msw))
    assert math.isclose(a["between_sd"], math.sqrt((msb - msw) / 3))


def test_reweighted_moves_mass_between_bins():
    rows = [{"logit": -2.0}] * 3 + [{"logit": 2.0}]
    vals = [[1.0]] * 3 + [[0.0]]
    even = FF.reweighted(rows, vals, 0.0, 1.0)["estimate"][0]
    assert math.isclose(even, 0.5, abs_tol=1e-9)       # symmetric target: the bins count alike
    low = FF.reweighted(rows, vals, -2.0, 0.3)
    assert low["estimate"][0] > 0.99
    # only the occupied bins' mass counts: [-2.25, -1.75) holds 2 * Phi(0.25 / 0.3) - 1
    assert math.isclose(low["coverage"], 2 * FF._ncdf(0.25 / 0.3) - 1, abs_tol=1e-6)


def test_verify_archive_checks_members_against_a_commit(tmp_path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    commit = head.stdout.strip()
    model = FF.git_show(commit, "submission/model.py")
    if model is None:
        pytest.skip("no submission/model.py at HEAD")
    good, bad = tmp_path / "good.zip", tmp_path / "bad.zip"
    for path, body in ((good, model), (bad, model + b"\n# edited\n")):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("model.py", body)
            z.writestr("prior.json", b"{}")
    ok = FF.verify_archive(str(good), commit)
    assert ok["all_tracked_members_match"] and ok["members"]["prior.json"]["source"] is None
    assert not FF.verify_archive(str(bad), commit)["all_tracked_members_match"]
