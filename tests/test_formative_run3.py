"""experiments/formative_run3.py: the record of formative run 3.

Run 3's table must give each pair's summary ALC from its budget rows and a
pair-mean ALC of 0.181653; the record must equal the record stage of
experiments/formative_feedback.py on runs 1 and 2 (whose platform scores it
reproduces); the overlap must find the one (subject, benchmark) pair run 3
shares with run 1; the placement must reproduce ship_confirm.json's own run-2
z values; the archive record must say whether the copy was downloaded back;
the default read of dist/ must write nothing unless it holds the selected
archive, and a stored record that verified it must not be replaced silently;
and the stored results file must be this table's record, with no reading.
"""
import json
import math
import os
import zipfile

import numpy as np
import pytest

from experiments import formative_feedback as FF
from experiments import formative_run3 as FR

ROOT = FR.ROOT


def _runs():
    out = {}
    for k, path in FR.FEEDBACK.items():
        with open(path) as f:
            out[k] = FF.parse(f.read())
    return out


def _table(rows):
    """A feedback table in the organisers' layout, the same rows in every section."""
    out = ["Formative feedback", ""]
    for head in ["ALC summary"] + [f"Label budget: {B}" for B in FF.BUDGETS]:
        out += [head, "", "subject_id  Benchmark  N  Brier  ECE", "---"]
        out += [f"{s}  {b}  {n}  {x:.6f}  {e:.6f}" for s, b, n, x, e in rows]
        out.append("")
    return "\n".join(out)


def test_run3_table_gives_its_summary_rows_and_alc():
    r3 = _runs()["run3"]
    assert len(r3) == 9
    s = FR.summarise(r3)
    assert s["counts"]["pairs"] == 9 and s["counts"]["benchmarks"] == 7 and s["counts"]["subjects"] == 9
    assert s["counts"]["pairs_per_benchmark"] == [2, 2, 1, 1, 1, 1, 1]
    assert s["per_pair_alc_max_abs_diff"] <= FR.PAIR_TOL
    assert s["per_pair_ece_alc_max_abs_diff"] <= FR.PAIR_TOL
    assert s["alc"]["recomputed_6dp"] == 0.181653
    assert math.isclose(s["alc"]["recomputed"], s["alc"]["from_budget_means"], abs_tol=1e-12)
    # pair-averaged, as runs 1 and 2 pinned; item weighting would differ
    assert round(s["alc"]["item_weighted_alternative"], 6) != s["alc"]["recomputed_6dp"]


@pytest.mark.parametrize("name", ["run1", "run2"])
def test_summarise_is_the_record_stage(name):
    pairs = _runs()[name]
    ref, mine = FF.summarise_run(name, pairs), FR.summarise(pairs)
    for f in FR.SHARED_FIELDS:
        assert ref[f] == mine[f], f
    for f, v in mine["alc"].items():
        assert ref["alc"][f] == v, f
    assert mine["alc"]["recomputed_6dp"] == FF.RUNS[name]["platform_alc"]


def test_overlap_with_runs_1_and_2():
    runs = _runs()
    o13 = FR.overlap_two("run1", runs["run1"], "run3", runs["run3"])
    assert o13["subjects"]["shared"] == ["subject_431933"]
    assert o13["pairs"]["shared"] == [["subject_431933", "benchmark_830060"]]
    assert o13["benchmarks"]["shared"] == 7
    (row,) = o13["paired_comparison"]
    assert row["n"] == [53, 53]
    want = float(np.dot(FR.W, row["run3_brier"]) - np.dot(FR.W, row["run1_brier"]))
    assert math.isclose(row["run3_minus_run1_alc"], want, abs_tol=1e-12)
    o23 = FR.overlap_two("run2", runs["run2"], "run3", runs["run3"])
    assert o23["subjects"]["shared"] == [] and o23["paired_comparison"] is None
    # the same sets as formative_feedback.overlap, only labelled by run
    ref = FF.overlap(runs["run1"], runs["run3"])
    assert ref["pairs"]["shared"] == o13["pairs"]["shared"]
    allr = FR.overlap_all(runs)
    assert allr["benchmark_ids_in_every_run"] == 7 and allr["distinct_benchmark_ids"] == 7
    assert allr["pair_appearances"] == 26 and allr["distinct_subjects"] == 25
    assert allr["subjects_in_more_than_one_run"] == {"subject_431933": ["run1", "run3"]}


def test_overlap_two_on_synthetic_tables():
    a = FF.parse(_table([("subject_1", "benchmark_1", 50, 0.2, 0.1)]))
    b = FF.parse(_table([("subject_1", "benchmark_1", 52, 0.1, 0.1),
                         ("subject_2", "benchmark_2", 44, 0.3, 0.1)]))
    ov = FR.overlap_two("x", a, "y", b)
    (row,) = ov["paired_comparison"]
    assert row["n"] == [50, 52] and math.isclose(row["y_minus_x_alc"], -0.1, abs_tol=1e-12)
    assert ov["benchmarks"] == {"x": 1, "y": 2, "shared": 1}


def test_letters_follow_run_1():
    runs = _runs()
    lt = FR.letters(runs["run1"])
    # docs/findings.md: A is run 1's first benchmark, G its last new one
    assert lt["benchmark_830060"] == "A" and lt["benchmark_800730"] == "B"
    assert lt["benchmark_119137"] == "G" and len(lt) == 7
    rows = FR.pair_rows(runs["run3"], lt)
    assert [r["label"] for r in rows] == [f"s{j}" for j in range(1, 10)]
    assert all(r["letter"] is not None for r in rows)


def _sc(mean_alc=0.2, sd_alc=0.02, mean_b=0.2, sd_b=0.04):
    regs, sds = {}, {}
    for r in ("tl", "r1b"):
        regs[r] = {"runs": 10, "mean_ALC": mean_alc, "sd_ALC": sd_alc, "mean_budgets": [mean_b] * 6,
                   "z_ALC": (0.19 - mean_alc) / sd_alc, "z_budgets": [(0.18 - mean_b) / sd_b] * 6,
                   "share_runs_at_or_above": 0.5}
        sds[r] = {"runs": 10, **{f"B{b}": sd_b for b in FF.BUDGETS}, "ALC": sd_alc}
    return {"run2_placement": {"run2": {"ALC": 0.19, "budgets": [0.18] * 6}, "regimes": regs},
            "single_run_sd": sds, "regimes": {r: {"label": r} for r in regs}, "code_gap": {}}


def test_zscores_and_placement_on_a_synthetic_regime():
    sc = _sc()
    z = FR.zscores(sc, [0.24] * 6, 0.25)
    assert math.isclose(z["tl"]["z_ALC"], 2.5) and all(math.isclose(x, 1.0) for x in z["tl"]["z_budgets"])
    sums = {"run2": {"budgets": {"brier_mean": [0.18] * 6}, "alc": {"recomputed": 0.19}},
            "run3": {"budgets": {"brier_mean": [0.24] * 6}, "alc": {"recomputed": 0.25}}}
    checks = FR.Checks()
    p = FR.placement(sc, sums, checks)
    assert not checks.failed()
    g = p["regimes"]["tl"]
    assert math.isclose(g["run3"]["z_ALC"], 2.5)
    assert math.isclose(g["run3_minus_run2"]["alc"], 0.06)
    assert math.isclose(g["run3_minus_run2"]["sd_of_difference_of_two_independent_runs"], math.sqrt(2) * 0.02)
    # a stored z that its own means and sds do not give fails the reading check
    sc["run2_placement"]["regimes"]["tl"]["z_ALC"] += 0.01
    bad = FR.Checks()
    FR.placement(sc, sums, bad)
    assert [c["check"] for c in bad.failed()] == [
        "ship_confirm tl: run 2's stored z_ALC recomputed from the stored means and sds"]


def test_placement_reproduces_ship_confirms_run2():
    if not os.path.exists(FR.SHIP_JSON):
        pytest.skip("no results/ship_confirm.json")
    with open(FR.SHIP_JSON) as f:
        sc = json.load(f)
    r2 = sc["run2_placement"]["run2"]
    z = FR.zscores(sc, r2["budgets"], r2["ALC"])
    for regime, g in sc["run2_placement"]["regimes"].items():
        assert abs(z[regime]["z_ALC"] - g["z_ALC"]) < 1e-4
        assert max(abs(a - b) for a, b in zip(z[regime]["z_budgets"], g["z_budgets"])) < 1e-4


def test_level_of_reads_model_py_without_importing():
    src = 'import os\nLEVEL = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}\nraise SystemExit\n'
    assert FR.level_of(src) == {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
    assert FR.level_of("X = 1\n") is None


def test_archive_record_says_whether_it_was_downloaded_back(tmp_path):
    missing = FR.archive_record(str(tmp_path / "none.zip"), False)
    assert missing["available"] is False and missing["downloaded_back"] is False
    assert "not downloaded back" in missing["provenance"]
    model = FF.git_show(FR.RUN3["commit"], "submission/model.py")
    if model is None:
        pytest.skip("commit 4d2cc4f not available")
    z = tmp_path / "copy.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("model.py", model)
        f.writestr("prior.json", b"{}")
    got = FR.archive_record(str(z), True, "downloaded from the submission page")
    assert got["available"] and got["downloaded_back"]
    assert got["provenance"].endswith("downloaded from the submission page")
    assert got["all_tracked_members_match"]
    assert got["equals_selected_archive"] is False          # not the selected archive's bytes
    assert got["level_in_model_py"] == FR.RUN3["level"]


def _default_archive_is_selected():
    return FR.sha256_file(FR.ARCHIVE) == FR.RUN3["archive_sha256"]


def _not_selected_zip(path):
    with zipfile.ZipFile(path, "w") as f:
        f.writestr("notes.txt", b"not the selected archive")
    return str(path)


def test_default_archive_checks_are_required(tmp_path, monkeypatch):
    # dist/ is mutable: without --archive, a missing or different file there writes nothing
    out = tmp_path / "run3.json"
    monkeypatch.setattr(FR, "ARCHIVE", str(tmp_path / "missing.zip"))
    with pytest.raises(SystemExit):
        FR.main(["--out", str(out)])
    assert not out.exists()
    monkeypatch.setattr(FR, "ARCHIVE", _not_selected_zip(tmp_path / "other.zip"))
    with pytest.raises(SystemExit):
        FR.main(["--out", str(out)])
    assert not out.exists()
    # a copy given with --archive is recorded whatever its bytes, its checks as warnings
    FR.main(["--out", str(out), "--archive", str(tmp_path / "other.zip"), "--archive-how", "test"])
    with open(out) as f:
        st = json.load(f)
    assert st["archive"]["downloaded_back"] and st["archive"]["equals_selected_archive"] is False
    arch = [c for c in st["checks"] if c["check"].startswith("archive:")]
    assert len(arch) == 3 and not any(c["required"] for c in arch)


def test_a_verified_archive_record_is_not_replaced_silently(tmp_path):
    out = tmp_path / "run3.json"
    stored = {"archive": {"available": True, "equals_selected_archive": True}}
    out.write_text(json.dumps(stored))
    other = _not_selected_zip(tmp_path / "other.zip")
    with pytest.raises(SystemExit):
        FR.main(["--out", str(out), "--archive", other])
    assert json.loads(out.read_text()) == stored
    FR.main(["--out", str(out), "--archive", other, "--replace-archive-record"])
    assert json.loads(out.read_text())["archive"]["equals_selected_archive"] is False
    assert FR.stored_archive_verified(str(tmp_path / "none.json")) is False


def test_main_writes_a_complete_record(tmp_path):
    if not _default_archive_is_selected():
        pytest.skip("dist/ does not hold the selected archive (4a882cc7...); the default read refuses")
    out = tmp_path / "run3.json"
    read_only = [FF.OUT, os.path.join(ROOT, "experiments", "formative_feedback.py"), FR.SHIP_JSON]
    before = [FF.sha256_file(p) for p in read_only if os.path.exists(p)]
    FR.main(["--out", str(out)])
    assert [FF.sha256_file(p) for p in read_only if os.path.exists(p)] == before   # only read
    with open(out) as f:
        st = json.load(f)
    assert all(c["ok"] for c in st["checks"] if c["required"])
    assert all(c["required"] for c in st["checks"] if c["check"].startswith("archive:"))
    assert FR.archive_verified(st["archive"]) and st["archive"]["downloaded_back"] is False
    assert st["run3"]["alc"]["recomputed_6dp"] == 0.181653
    assert st["run3"]["platform_alc"] is None
    assert [r["run"] for r in st["three_runs"]["table"]] == [1, 2, 3]
    assert set(st["placement"]["regimes"]) >= {"tl", "r1b", "r1p"}
    assert "no_reading_or_tuning" in st and "reading" not in st and "decision" not in st
    assert st["provenance"]["inputs_sha256"]["results/formative/run3.txt"] == FF.sha256_file(FR.FEEDBACK["run3"])


def test_stored_record_is_this_table():
    if not os.path.exists(FR.OUT):
        pytest.skip("no results/formative_run3.json")
    with open(FR.OUT) as f:
        st = json.load(f)
    assert all(c["ok"] for c in st["checks"] if c["required"])
    assert st["provenance"]["inputs_sha256"]["results/formative/run3.txt"] == FF.sha256_file(FR.FEEDBACK["run3"])
    # the stored record verified the selected archive, as docs/findings.md says
    assert FR.archive_verified(st["archive"]) and st["archive"]["all_tracked_members_match"]
    assert st["archive"]["sha256"] == FR.RUN3["archive_sha256"]
    s = FR.summarise(_runs()["run3"])
    assert st["run3"]["alc"] == s["alc"] and st["run3"]["budgets"] == s["budgets"]
    # nothing is read or tuned from run 3
    assert "reading" not in st and "level_distribution" not in json.dumps(st)

