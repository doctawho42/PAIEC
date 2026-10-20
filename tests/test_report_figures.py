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


# v2 (2026-10-14): the hidden runs read through Brier = ECE^2 + p(1-p), the regimes
# checked against them, and gate_curve split into (a, b) and text_signals (v1's (c))

#: the results files whose content is formative feedback or a reading of it (App G.6
#: declares each use); the figures read feedback from these files only
FORMATIVE_FILES = {"results/formative_feedback.json", "results/formative_run3.json",
                   "results/hidden_runs.json"}
#: the figures that read them, and nothing else does
FORMATIVE_FIGURES = {"learning_curves", "formative_runs", "hidden_decomposition", "regime_check"}
#: other results files that figures read hold copies or readings of the feedback in these
#: blocks (key paths); no figure reads them, which the strip test below checks
FORMATIVE_COPIES = {
    "results/ship_confirm.json": [("run2_placement",)],
    "results/testlike_check.json": [("summary", "feedback"), ("summary", "tune"),
                                    ("summary", "level_identification")],
    "results/level_calibration.json": [("summary", "feedback"), ("summary", "ppc")],
    "results/level_audit.json": [("feedback",),
                                 ("mild", "feedback-matched estimate, MILD and AGGR on one pool")],
}
#: and testlike_check.json's checks against the feedback, under every regime it scored
TESTLIKE_FEEDBACK_KEYS = ("ppc", "distance", "distance_with_B31", "distance_B3_B7_B15")


def test_only_declared_figures_read_formative_feedback(committed):
    for name, f in committed["figures"].items():
        reads = FORMATIVE_FILES & set(f["inputs"])
        assert bool(reads) == (name in FORMATIVE_FIGURES), (name, sorted(reads))
        assert "results/pool_eval.json" not in f["inputs"], name


def test_no_figure_reads_the_feedback_copies_in_other_files():
    """Every figure builds with the feedback's copies stripped from the other results files
    it reads, so the four declared figures are the only readers, through FORMATIVE_FILES."""
    import copy
    import matplotlib.pyplot as plt
    inp = RF.Inputs(ROOT)
    for rel, paths in FORMATIVE_COPIES.items():
        d = copy.deepcopy(inp.load(rel))
        for path in paths:
            node = d
            for k in path[:-1]:
                node = node[k]
            assert path[-1] in node, (rel, path)  # the block exists, so stripping it tests something
            del node[path[-1]]
        if rel == "results/testlike_check.json":
            stripped = 0
            for group in ("check", "r1", "sensitivity"):
                for block in d["summary"][group].values():
                    for k in TESTLIKE_FEEDBACK_KEYS:
                        stripped += block.pop(k, None) is not None
            assert stripped >= len(TESTLIKE_FEEDBACK_KEYS), stripped
        inp.cache[rel] = d
    with plt.rc_context(RF.RC):
        for _name, fn in RF.FIGURES:
            plt.close(fn(inp))


def test_gate_curve_is_two_panels_and_text_signals_holds_its_old_panel_c(committed):
    import matplotlib.pyplot as plt
    inp = RF.Inputs(ROOT)
    with plt.rc_context(RF.RC):
        assert len(RF.fig_gate_curve(inp).axes) == 2
        assert len(RF.fig_text_signals(inp).axes) == 1
    plt.close("all")
    gate, text = committed["figures"]["gate_curve"]["inputs"], committed["figures"]["text_signals"]["inputs"]
    assert "results/emb_transfer.json" not in gate and "results/emb_transfer.json" in text
    assert "results/harness_thresholds.json" in gate and "results/harness_thresholds.json" in text


def test_hidden_decomposition_matches_its_sources():
    inp = RF.Inputs(ROOT)
    d = RF.hidden_decomposition_data(inp)
    h = inp.load("results/hidden_runs.json")
    dec = h["decomposition"]
    assert d["curves"]["run1"]["pairs"] == 9 and d["curves"]["hier_pooled"]["pairs"] == 17
    for key, c in d["curves"].items():
        assert np.allclose(c["excess"], dec[key]["excess_by_budget"], atol=0)
        assert np.allclose(c["se"], dec[key]["pair_se"]["excess_by_budget_pair_se"], atol=0)
    # the pooled curve is the main text's 0.0662, 0.0208, 0.0225, 0.0048, 0.0026, 0.0036
    assert [round(v, 4) for v in d["curves"]["hier_pooled"]["excess"]] == \
        [0.0662, 0.0208, 0.0225, 0.0048, 0.0026, 0.0036]
    # ideal learner on the same 17 pairs: lower roots under run 1's reading, and the
    # prior-weighted band over the three priors, whose ALC - v spans 0.0250 to 0.0322
    lo, hi = d["ideal"]["band"]
    assert d["ideal"]["pairs"] == 17 and (lo <= hi).all()
    assert RF.alc(d["ideal"]["lower"]) == pytest.approx(
        h["ideal"]["summary_hier"]["lower_roots_team_prior"], abs=1e-12)
    band = [RF.alc(h["ideal"][p + " | runs (2, 3)"]["mix"]) for p in RF.IDEAL_PRIORS]
    assert [min(band), max(band)] == pytest.approx(h["ideal"]["summary_hier"]["prior_weighted_range"],
                                                   abs=1e-12)
    # bars: the three terms add up to ALC - v, hidden from the file, replica derived
    hidden = [b for b in d["bars"] if b["hidden"]]
    assert [b["label"] for b in hidden] == ["run 1", "run 2", "run 3"]
    for b in d["bars"]:
        assert sum(b["terms"]) == pytest.approx(b["total"], abs=1e-12), b["label"]
    for b, r in zip(hidden, ("run1", "run2", "run3")):
        assert b["total"] == dec[r]["alc_minus_v"]
        assert b["whisker"] == pytest.approx(h["vadjusted"][r]["se"], abs=1e-15)
        lo_, hi_ = b["item"]
        assert 0 < lo_ < hi_ and (lo_, hi_) == tuple(h["item_level"][r]["v_minus_item_oracle_range"])
    rep = [b for b in d["bars"] if not b["hidden"]]
    assert sorted({b["label"] for b in rep}) == ["R1-bf", "reading", "tuned"]
    for b in rep:
        n = {"tuned": "TUNED", "reading": "READING", "R1-bf": "R1B"}[b["label"]]
        arm = h["replica"]["noise"][n]["ship" if b["arm"] == "ship" else "legacy"]
        assert b["total"] == arm["alc_minus_v"] and b["whisker"] == arm["alc_minus_v_sd"]
    # Table 1's values, as the main text quotes them
    assert [round(b["total"], 4) for b in hidden] == [0.0577, 0.0130, 0.0208]


def test_regime_check_matches_its_sources():
    inp = RF.Inputs(ROOT)
    d = RF.regime_check_data(inp)
    rs = inp.load("results/regime_sensitivity.json")["summary"]
    h = inp.load("results/hidden_runs.json")
    assert d["order"][:5] == sorted(RF.TEST_LIKE, key=lambda r: rs["realised"][r]["mean"])
    assert d["order"][5:] == ["R1B", "R1P"]
    for r in d["order"]:
        for c in RF.RC_CONFIGS:
            ex = rs["vs_ship"][r][c]["exact"]
            assert (d["rows"][r][c]["D"], d["rows"][r][c]["se"]) == (ex["D"], ex["cluster_se"])
    # Table 3's LegacyP row, comparator minus hier-ship
    assert [round(d["rows"][r]["legacy"]["D"], 4) for r in
            ("TUNED", "AUDIT", "MIXTURE", "READING", "FLAT", "R1B", "R1P")] == \
        [0.0434, 0.0313, 0.0263, 0.0248, 0.0200, 0.0002, 0.0039]
    # held-out TL runs 200-299: stored as hier-ship minus LegacyP, drawn with the sign flipped;
    # hidden_runs.json carries the same pair
    ho = d["held_out"]
    assert ho["D"] == -h["main"]["held_out_tl_shipped_minus_legacy"] == 0.041911
    assert ho["se"] == h["main"]["held_out_tl_cluster_se"] == 0.003736
    assert round(ho["level"], 2) == -1.29
    hid = d["hidden"]
    assert (round(hid["D"], 4), round(hid["se"], 4)) == (0.0405, 0.0191)
    assert hid["D"] == h["vadjusted"]["agreement"]["hidden_legacy_minus_hier"]
    # placement z: raw ALC and ALC - v, the two stored ways
    for who in ("run1 (LegacyP)", "runs 2+3 (hier)"):
        for r in d["order"]:
            raw, adj = d["place"][who][r]
            assert raw == h["replica"]["placement_raw"][r][who]["z_alc"]
            assert adj == h["replica"]["placement_z"][r][who]["z_alc_minus_v"]
    assert [round(d["place"]["run1 (LegacyP)"][r][1], 2) for r in ("R1B", "R1P")] == [3.78, 2.93]


# type size (author decision, 2026-10-20): the main text's figures carry no type under 6 pt
# at print size (they are drawn at the text width and placed at it), and they do not grow,
# since the main text is at its 8-page limit


def test_drawn_text_records_what_is_drawn():
    """The measure itself: tick labels, legend entries, annotations and axis labels are
    recorded at their sizes, a mathtext subscript at 0.7 of its label's, and a hidden text or
    a tick outside the view is not recorded."""
    import matplotlib.pyplot as plt
    with plt.rc_context(RF.RC):
        fig, ax = plt.subplots(figsize=(RF.WIDTH, 2.0))
        ax.plot([0, 1], [0, 1], label="entry")
        ax.legend(fontsize=5.5)
        ax.set_xticks([0, 0.5, 1, 2])  # 2 lies outside the view and is not drawn
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6.5)
        ax.annotate("note", (0.5, 0.5), fontsize=5.9)
        ax.set_ylabel("B$_k$ − V", fontsize=7.0)
        ax.text(0.2, 0.2, "hidden", fontsize=3.0, visible=False)
        t = RF.drawn_text(fig)
        plt.close(fig)
    by = {s: (size, glyph) for size, glyph, s in t}
    assert by["entry"] == (5.5, 5.5) and by["note"] == (5.9, 5.9)
    assert by["0.5"] == (6.5, 6.5) and "2.0" not in by and "hidden" not in by
    assert by["B$_k$ − V"][0] == 7.0 and by["B$_k$ − V"][1] == pytest.approx(4.9, abs=1e-6)


def test_main_text_figures_are_the_placement_tables():
    """RF.MAIN_TEXT is exactly the drawn figures that docs/report/figures.md places in the main
    text (draft figures numbered without an appendix letter)."""
    import re
    md = (ROOT / "docs" / "report" / "figures.md").read_text()
    rows = re.findall(r"^\| Figure ([A-Z]?)\d+ \| [^|]+ \| D\d+ \| `([^`]+)` \|", md, re.M)
    assert rows and set(RF.MAIN_TEXT) == {stem for letter, stem in rows if not letter}
    assert RF.MIN_FONT_PT == 6.0


@pytest.mark.parametrize("name", sorted(RF.MAIN_TEXT))
def test_main_text_figure_has_no_type_under_6pt(name):
    import matplotlib.pyplot as plt
    fn = dict(RF.FIGURES)[name]
    with plt.rc_context(RF.RC):
        fig = fn(RF.Inputs(ROOT))
        w, h = fig.get_size_inches()
        t = RF.drawn_text(fig)
        plt.close(fig)
    assert w == RF.WIDTH and h <= RF.MAIN_TEXT[name], (w, h)  # the text width, and no taller
    assert len(t) > 40  # tick labels, legend entries, annotations and bar labels, line by line
    small = sorted({(size, s) for size, _, s in t if size < RF.MIN_FONT_PT})
    assert not small, small
    # nor a smaller glyph: a mathtext sub- or superscript is set at 0.7 of its label's size
    scripts = sorted({(glyph, s) for _, glyph, s in t if glyph < RF.MIN_FONT_PT})
    assert not scripts, scripts


def test_manifest_records_type_sizes(committed):
    for name, f in committed["figures"].items():
        w, h = f["size_in"]
        assert 0 < w <= RF.WIDTH and h > 0, name
        assert 0 < f["text_pt"]["min_glyph_pt"] <= f["text_pt"]["min_pt"], name
    for name, hmax in RF.MAIN_TEXT.items():
        f = committed["figures"][name]
        assert f["text_pt"]["min_pt"] >= RF.MIN_FONT_PT and f["text_pt"]["lines_under_min"] == 0
        assert f["text_pt"]["min_glyph_pt"] >= RF.MIN_FONT_PT, name  # scripts included
        assert f["size_in"][0] == RF.WIDTH and f["size_in"][1] <= hmax, name
