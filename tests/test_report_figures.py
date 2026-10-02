"""tools/report_figures.py: every figure builds from the committed results files
alone, reads nothing else, is deterministic, and the committed figures under
docs/report/fig/ match their manifest (inputs, script and outputs). No data
download is needed; matplotlib is (skipped without it)."""
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "fig"


def _load():
    spec = importlib.util.spec_from_file_location("report_figures", ROOT / "tools" / "report_figures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["report_figures"] = mod
    spec.loader.exec_module(mod)
    return mod


RF = _load()
NAMES = [n for n, _ in RF.FIGURES]


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("fig")
    return out, RF.build(out)


@pytest.fixture(scope="module")
def committed():
    mp = FIG / "manifest.json"
    if not mp.exists():
        pytest.fail("docs/report/fig/manifest.json missing: run python tools/report_figures.py")
    return json.loads(mp.read_text())


def test_every_figure_builds_from_results_only(fresh):
    out, man = fresh
    assert list(man["figures"]) == NAMES
    for name, f in man["figures"].items():
        assert f["inputs"], name
        for rel in f["inputs"]:
            assert RF.Inputs.allowed(rel), (name, rel)
            assert rel.startswith("results/") and rel.endswith(".json")
            assert (ROOT / rel).exists(), rel
        assert sorted(f["outputs"]) == sorted([f"{name}.png", f"{name}.svg"])
        svg = out / f"{name}.svg"
        root = ET.parse(svg).getroot()
        assert root.tag.endswith("svg")
        assert b"<dc:date>" not in svg.read_bytes()  # no timestamp in the file
        png = (out / f"{name}.png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000
        assert b"Software" not in png[:2000]


def test_inputs_refuse_anything_but_results_json(tmp_path):
    inp = RF.Inputs(ROOT)
    for bad in ("data/x.json", "results/sub/x.json", "results/inventory.csv",
                "../results/ship_confirm.json", "/etc/passwd", "docs/findings.md",
                "results/formative/run1.txt"):
        with pytest.raises(ValueError):
            inp.load(bad)
    inp.begin()
    inp.load("results/ship_confirm.json")
    assert list(inp.record()) == ["results/ship_confirm.json"]


def test_build_is_deterministic(tmp_path):
    a = RF.build(tmp_path / "a", only=["item_gap", "level_surface"])
    b = RF.build(tmp_path / "b", only=["item_gap", "level_surface"])
    assert a["figures"] == b["figures"]
    for n in ("item_gap.svg", "level_surface.svg", "item_gap.png"):
        assert (tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes()


def test_committed_figures_match_manifest(committed):
    assert committed["script"] == RF.SCRIPT
    assert committed["script_sha256"] == RF.sha256_file(ROOT / RF.SCRIPT), \
        "tools/report_figures.py changed: rerun it"
    assert list(committed["figures"]) == NAMES
    for name, f in committed["figures"].items():
        for rel, sha in f["inputs"].items():
            assert RF.Inputs.allowed(rel)
            assert RF.sha256_file(ROOT / rel) == sha, f"{rel} changed since {name} was drawn"
        for fn, sha in f["outputs"].items():
            assert (FIG / fn).exists(), fn
            assert RF.sha256_file(FIG / fn) == sha, fn


def test_fresh_build_reproduces_committed(fresh, committed):
    _, man = fresh
    for name in NAMES:
        assert man["figures"][name]["inputs"] == committed["figures"][name]["inputs"], name
    if man["versions"]["matplotlib"] != committed["versions"]["matplotlib"]:
        pytest.skip("another matplotlib: bytes are only comparable on the same version")
    for name in NAMES:
        assert man["figures"][name]["outputs"] == committed["figures"][name]["outputs"], name


def test_check_reports_no_problem():
    assert RF.check(FIG, rebuild=False) == []


# what the panels plot is what the results files hold


def test_learning_curve_series_match_their_sources():
    inp = RF.Inputs(ROOT)
    ser = RF.learning_curve_series(inp)
    sc = inp.load("results/ship_confirm.json")
    f3 = inp.load("results/formative_run3.json")
    for key in ser:
        pl = sc["run2_placement"]["regimes"][key]
        assert np.allclose(ser[key]["ship"], pl["mean_budgets"], atol=1e-6)
        assert np.allclose(ser[key]["sd"], f3["placement"]["regimes"][key]["sd_budgets"],
                           atol=1e-6)
        assert ser[key]["ship_runs"] == pl["runs"]
        assert "legacy" in ser[key] and "smoothed" in ser[key]
    runs, pairs = RF.formative_runs(inp)
    for row in f3["three_runs"]["table"]:
        assert np.allclose(runs[row["run"]], row["brier_mean"], atol=1e-9)
        assert len(pairs[row["run"]]) == row["pairs"]
        assert RF.alc(runs[row["run"]]) == pytest.approx(row["alc_recomputed"], abs=1e-9)


def test_surface_marks_and_sigma_strip():
    inp = RF.Inputs(ROOT)
    d = RF.surface_data(inp)
    assert d["marks"]["ship"] == "hier G mu0=-2.50 sm=2.50 as=0.50"
    assert d["marks"]["aggr"] == "hier G mu0=-3.00 sm=2.50 as=0.25"
    assert d["marks"]["chosen"] == "hier G mu0=-3.50 sm=2.50 as=0.25"
    lc = inp.load("results/level_calibration.json")["summary"]
    i, j = RF.GRID_AS.index("0.5"), d["mu"].index(-2.5)
    assert d["alc"][i, j] == pytest.approx(
        lc["selection"]["configs"]["hier G mu0=-2.50 sm=2.50 as=0.50"]["ALC"][0], abs=1e-12)
    # the strip's claim: sigma_mu 2.5 is the best width in every scored cell
    assert all((v > 0).all() for v in d["sm_gap"].values())
    # the shipped cell's guard is the 100-run one the level audit scored on the same runs
    # (the audit's numbers for the neighbouring config equal the level calibration's)
    la = inp.load("results/level_audit.json")["mild"]
    got = {r: la[RF.AUDIT_R1[r]]["vs PRED"][d["marks"]["ship"]]["minus ref (run / cluster / stratified SE)"][0]
           for r in ("r1b", "r1p")}
    assert d["guard_src"][i, j] == "audit" and d["guard"][i, j] == max(got.values())
    for r in ("r1b", "r1p"):
        aud = la[RF.AUDIT_R1[r]]["vs PRED"][d["marks"]["aggr"]]
        assert aud["runs"] == 100 == lc["r1"][r][d["marks"]["aggr"]]["runs"]
        assert aud["minus ref (run / cluster / stratified SE)"][0] == pytest.approx(
            lc["r1"][r][d["marks"]["aggr"]]["diff"]["mean"], abs=5e-5)
    assert (d["guard_src"] == "audit").sum() == 1


def test_onepl_is_not_called_a_1pl():
    # P1a's onepl is hier with its subject prior off, not P1.9's plain 1PL (rasch)
    assert "1PL" not in RF.NAMES["onepl"] and "1PL" not in RF.RS_NAMES["onepl"]


def test_gate_and_covariates():
    inp = RF.Inputs(ROOT)
    t = RF.gate_table(inp)
    assert t["r"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    assert t["gate"]["tl"] == -0.002
    assert all(len(r) == 8 for r in t["lines"]["transferred nested"]["reps"])
    pts = RF.covariate_points(inp)
    assert {p["family"] for p in pts} == {f for f, *_ in RF.COVARIATES}
    # none of the measured covariates reaches the gate on its nested line
    assert min(p["est"] for p in pts) > t["gate"]["tl"]
    tp = RF.transfer_points(inp)
    assert all(q["ci"][0] <= q["est"] <= q["ci"][1] for q in tp)


def test_regime_order_and_rows():
    inp = RF.Inputs(ROOT)
    d = RF.regime_sensitivity_data(inp)
    tl = [r for r in d["order"] if not r.startswith("R1")]
    assert [d["realised"][r]["mean"] for r in tl] == sorted(d["realised"][r]["mean"] for r in tl)
    assert d["order"][-2:] == ["R1B", "R1P"]
    for r in d["order"]:
        for c in RF.RS_CONFIGS:
            v = d["rows"][r][c]
            assert v["U95"] == pytest.approx(v["D"] + 1.96 * v["se"], abs=1e-9)
