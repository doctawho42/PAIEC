"""experiments/factorial_rs.py (W3): the calibration-by-model factorial on the
regime-sensitivity runs.

On synthetic rows (no download needed): every two-cell contrast is RS's own
paired difference (RegimeRows.diff and its Block table) bit for bit; the
interaction is the difference of the simple effects either way round, the main
effects their means; additive effects give an interaction of zero and a known
extra calibration effect on hier gives it back. On the stored rows (skipped
when data/regime_sensitivity_rows/ is absent): one regime's cell means,
recomputed here from the row files with numpy and level_calibration.Boot alone,
equal results/regime_sensitivity.json, and so does the interaction derived
from them. The stored results file (skipped when not written) is this script's
output on the committed RS results, passed every check, and carries no
subject id. Without rows, its numbers are pinned: each numeric block by the
sha256 of its canonical JSON, the interaction's D and cluster SE per regime by
value, every contrast's D as the signed sum of RS's stored differences, the
summary as the summary of the stored contrasts; with the rows, every block
recomputes from them exactly.
"""
import hashlib
import json
import math
import os

import numpy as np
import pytest

import experiments.level_calibration as LC
from experiments import factorial_rs as FR
from experiments import regime_sensitivity as RS

W6 = np.asarray(RS.WEIGHTS, float)
#: the regime whose cell means are recomputed from the rows
REGIME = "READING"


def synthetic_rows(runs=12, seed=0, effects=None):
    """Rows as RS's score stage writes them, every config, runs of 3 to 7 pairs over the
    four multi-subject parents. effects: config -> additive Brier shift (all budgets);
    absent configs get a small random shift."""
    rng = np.random.default_rng(seed)
    shift = {c: rng.normal(0, 0.01) for c in RS.CONFIGS}
    shift.update(effects or {})
    rows = {}
    for i in range(runs):
        k = int(rng.integers(3, 8))
        meta = [{"bench": f"{q}::g{j}", "parent": q, "subject": f"s{int(rng.integers(0, 5))}", "eval": 50,
                 "p": 0.3, "logit": float(rng.normal(-1, 2))}
                for j, q in enumerate(RS.PARENTS[int(rng.integers(0, 4))] for _ in range(k))]
        base = rng.uniform(0.05, 0.3, (k, 6))
        res = {}
        for c in RS.CONFIGS:
            noise = 0.0 if effects else rng.normal(0, 0.01, (k, 6))
            b = base + shift[c] + noise
            res[c] = {"b": b.tolist(), "ece": rng.uniform(0, 0.2, (k, 6)).tolist(),
                      "q0": rng.uniform(0, 1, k).tolist(), "calls": 100, "mean_s": 0.01,
                      "max_s": 0.05, "failures": 0, "unconverged": 0, "secs": 1.0}
        rows[i] = {"meta": meta, "res": res}
    return rows


def contrasts_of(reg):
    return {name: reg.contrast(w) for name, w in FR.CONTRASTS.items()}


# --- synthetic rows ------------------------------------------------------------------------------

def test_two_cell_contrasts_are_rs_paired_differences():
    reg = FR.Regime("TUNED", synthetic_rows())
    for name, w in FR.CONTRASTS.items():
        if len(w) != 2:
            continue
        (x, a), (ref, b) = w
        assert (a, b) == (1.0, -1.0), name
        got, want = reg.contrast(w), reg.G.diff(x, ref)
        for k in ("D", "run_se", "cluster_se", "strat_se", "pct", "share_runs_x_better"):
            g = got["share_runs_negative"] if k == "share_runs_x_better" else got[k]
            assert g == want[k], (name, k)
        t = reg.G.table(x, ref)
        assert [r["diff"] for r in got["budgets"]] == [r["diff"] for r in t["budgets"]]
        assert [r["cluster_se"] for r in got["budgets"]] == [r["cluster_se"] for r in t["budgets"]]
        assert got["per_parent"] == t["per_parent"] and got["parent_level"] == t["parent_level"]


def test_interaction_and_main_effects_are_the_factorial_algebra():
    c = contrasts_of(FR.Regime("AUDIT", synthetic_rows(seed=3)))
    I = c["interaction"]["D"]
    assert math.isclose(I, c["calibration_in_hier"]["D"] - c["calibration_in_smooth"]["D"], abs_tol=1e-15)
    assert math.isclose(I, c["model_at_calibrated"]["D"] - c["model_at_default"]["D"], abs_tol=1e-15)
    assert math.isclose(c["calibration_main"]["D"],
                        (c["calibration_in_hier"]["D"] + c["calibration_in_smooth"]["D"]) / 2, abs_tol=1e-15)
    assert math.isclose(c["model_main"]["D"],
                        (c["model_at_default"]["D"] + c["model_at_calibrated"]["D"]) / 2, abs_tol=1e-15)
    assert c["interaction"]["cluster_se"] > 0 and c["interaction"]["run_se"] > 0
    assert c["interaction"]["ci95_cluster"] == pytest.approx(
        [I - 1.96 * c["interaction"]["cluster_se"], I + 1.96 * c["interaction"]["cluster_se"]])


@pytest.mark.parametrize("extra", [0.0, -0.0123])
def test_interaction_recovers_a_known_extra_effect(extra):
    """Calibration lowers every Brier by 0.01 in both models, and by `extra` more in
    hier: the ALC weights sum to 1, so the interaction is `extra` on every pair, with
    no spread."""
    eff = {"smooth": 0.0, "smcal": -0.01, "eb_fit": 0.02, "ship": 0.02 - 0.01 + extra}
    assert math.isclose(float(W6.sum()), 1.0, abs_tol=1e-15)
    c = contrasts_of(FR.Regime("FLAT", synthetic_rows(seed=5, effects=eff)))
    assert c["interaction"]["D"] == pytest.approx(extra, abs=1e-12)
    assert c["interaction"]["cluster_se"] == pytest.approx(0.0, abs=1e-12)
    assert c["calibration_in_smooth"]["D"] == pytest.approx(-0.01, abs=1e-12)
    assert c["model_at_default"]["D"] == pytest.approx(0.02, abs=1e-12)
    for b in c["interaction"]["budgets"]:
        assert b["diff"] == pytest.approx(extra, abs=1e-6)


def test_contrasts_cover_the_outline_cells():
    assert set(FR.CELLS.values()) == {"smooth", "smcal", "eb_fit", "ship", "legacy"}
    assert set(FR.FACTORIAL.values()) == {"smooth", "smcal", "eb_fit", "ship"}
    assert set(FR.CELLS.values()) <= set(RS.CONFIGS)
    for w in FR.CONTRASTS.values():
        assert math.isclose(sum(x for _, x in w), 0.0, abs_tol=1e-15)


# --- the stored rows -------------------------------------------------------------------------------

def _rows_or_skip(R):
    if not all(os.path.exists(RS.row_path(R, i)) for i in range(RS.N_RUNS[R])):
        pytest.skip("data/regime_sensitivity_rows/ not present")
    if not os.path.exists(RS.OUT):
        pytest.skip("results/regime_sensitivity.json not present")
    rows = {}
    for i in range(RS.N_RUNS[R]):
        with open(RS.row_path(R, i)) as f:
            rows[i] = json.load(f)
    with open(RS.OUT) as f:
        return rows, json.load(f)["summary"]


def test_cell_means_from_the_rows_match_regime_sensitivity():
    """One regime's five cells recomputed from the row files with numpy and
    level_calibration.Boot only, against results/regime_sensitivity.json."""
    rows, summary = _rows_or_skip(REGIME)
    ids = sorted(rows)
    metas = [rows[i]["meta"] for i in ids]
    boot = LC.Boot(metas, RS.BOOTS, RS.BOOT_SEED)
    alc = {}
    for c in FR.CELLS.values():
        B = [np.asarray(rows[i]["res"][c]["b"], float) for i in ids]
        alc[c] = [b @ W6 for b in B]
        run_alc = np.array([np.mean(a) for a in alc[c]])
        want = summary["configs"][REGIME][c]
        assert want["runs"] == len(ids) == RS.N_RUNS[REGIME]
        assert math.isclose(run_alc.mean(), want["ALC"], abs_tol=1e-12), c
        assert math.isclose(run_alc.std(ddof=1) / math.sqrt(len(ids)), want["ALC_run_se"], abs_tol=1e-12), c
        assert math.isclose(boot.se(alc[c])[0], want["ALC_cluster_se"], abs_tol=1e-12), c
        budgets = np.mean([b.mean(0) for b in B], 0)
        assert np.allclose(budgets, want["budgets"], rtol=0, atol=1e-12), c
        q0 = np.mean([np.mean(rows[i]["res"][c]["q0"]) for i in ids])
        assert math.isclose(q0, want["q0"], abs_tol=1e-12), c
    # the script's own cells on the same rows
    reg = FR.Regime(REGIME, rows)
    for c in FR.CELLS.values():
        got = reg.cell(c)
        for k in ("ALC", "ALC_run_se", "ALC_cluster_se", "q0"):
            assert got[k] == summary["configs"][REGIME][c][k], (c, k)
    # the interaction from the same per pair ALCs, against RS's stored differences
    d = [s - e - m + o for s, e, m, o in zip(alc["ship"], alc["eb_fit"], alc["smcal"], alc["smooth"])]
    I = float(np.mean([np.mean(x) for x in d]))
    ex = {X: summary["vs_ship"][REGIME][X]["exact"]["D"] for X in ("eb_fit", "smcal", "smooth")}
    assert math.isclose(I, -ex["eb_fit"] - ex["smcal"] + ex["smooth"], abs_tol=1e-12)
    got = reg.contrast(FR.CONTRASTS["interaction"])
    assert math.isclose(got["D"], I, abs_tol=1e-15)
    assert math.isclose(got["cluster_se"], boot.se(d)[0], abs_tol=1e-15)


# --- the stored results file -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stored():
    if not os.path.exists(FR.OUT):
        pytest.skip("results/factorial_rs.json not written")
    with open(FR.OUT) as f:
        return json.load(f)


def test_stored_file_is_this_scripts_output(stored):
    prov = stored["provenance"]
    assert prov["script"] == "experiments/factorial_rs.py"
    assert prov["script_sha256"] == FR.sha256(os.path.abspath(FR.__file__))
    assert prov["inputs_sha256"]["results/regime_sensitivity.json"] == FR.sha256(RS.OUT)
    for k in ("git_head", "command", "when_utc", "wall_s", "rows"):
        assert prov.get(k) not in (None, ""), k
    assert all(c["ok"] for c in stored["checks"]), [c["name"] for c in stored["checks"] if not c["ok"]]
    assert {c["name"] for c in stored["checks"]} >= {
        "rows_complete", "rows_scored_script", "cells_equal_rs_configs", "pairs_equal_rs_exact",
        "pairs_equal_rs_tables", "interaction_equals_rs_stored_differences", "factorial_algebra"}


def test_stored_contrasts_against_the_committed_rs_results(stored):
    """Without rows: the stored two-cell contrasts are RS's stored ones with the stated
    sign, and the interaction is their signed sum."""
    with open(RS.OUT) as f:
        summary = json.load(f)["summary"]
    for R in RS.REGIMES:
        c = stored["contrasts"][R]
        ex = {X: summary["vs_ship"][R][X]["exact"] for X in ("eb_fit", "smcal", "smooth", "legacy")}
        assert c["calibration_in_hier"]["D"] == -ex["eb_fit"]["D"]
        assert c["calibration_in_hier"]["cluster_se"] == ex["eb_fit"]["cluster_se"]
        assert c["model_at_calibrated"]["D"] == -ex["smcal"]["D"]
        assert c["legacy_minus_ship"]["D"] == ex["legacy"]["D"]
        assert c["legacy_minus_ship"]["cluster_se"] == ex["legacy"]["cluster_se"]
        assert math.isclose(c["interaction"]["D"], -ex["eb_fit"]["D"] - ex["smcal"]["D"] + ex["smooth"]["D"],
                            abs_tol=1e-12)
        t = summary["vs_smooth"][R]["smcal"]["diff"]
        assert abs(c["calibration_in_smooth"]["D"] - t["mean"]) <= 5e-7
        assert abs(c["calibration_in_smooth"]["cluster_se"] - t["cluster_se"]) <= 5e-7
        for k, cell in FR.CELLS.items():
            assert stored["cells"][R][cell]["ALC"] == summary["configs"][R][cell]["ALC"]


#: the interaction (ship - eb_fit - smcal + smooth) per regime as results/factorial_rs.json
#: stores it: D and the paired cluster SE, full precision. test_stored_file_recomputes_from_
#: the_rows checks these against the rows where they are on disk; here they pin the file.
INTERACTION = {
    "TUNED": (-0.012465708266796067, 0.002099044675367138),
    "READING": (-0.015492729512395841, 0.0014737712761116618),
    "AUDIT": (-0.01334713854815019, 0.0017334609958437662),
    "MIXTURE": (-0.012461005856487769, 0.0019007031680639919),
    "FLAT": (-0.01401285518688281, 0.001492587495692046),
    "R1B": (-0.0022596717788615057, 0.00261605659841182),
    "R1P": (0.0018584116676013563, 0.0022718569875429385),
}
#: sha256 of each numeric block of the stored file as canonical JSON (sorted keys, no
#: whitespace): any change to a stored number fails. Re-pin only after a re-run that
#: test_stored_file_recomputes_from_the_rows passes on the rows.
BLOCK_SHA256 = {
    "cells": "d84f395d4934b55a08734986441daca78a58a482400b93bb4a94ac3049bd2443",
    "contrasts": "98c3ab32118d5733b9793463a8f76949a3f3e4fd8e94a72256a661f4427d6a21",
    "summary": "245970d14167226b8686e837ad8642f37b1ff2e1444d5624b13211fb4e1f51cc",
    "outline_check": "2ca0e87cb27d8822563d6589a62655d1ac9d22ecb66756c2ad241d5ad208ce07",
}


def canonical_sha256(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_stored_numeric_blocks_are_pinned(stored):
    for k, want in BLOCK_SHA256.items():
        assert canonical_sha256(stored[k]) == want, k


def test_stored_interaction_cluster_se_is_pinned(stored):
    """Without rows: the interaction's D and cluster SE in every regime, and the CI and z
    derived from them, in the contrasts and in the summary."""
    assert set(stored["contrasts"]) == set(INTERACTION) == set(RS.REGIMES)
    for R, (D, se) in INTERACTION.items():
        c = stored["contrasts"][R]["interaction"]
        assert c["D"] == D and c["cluster_se"] == se, R
        assert c["ci95_cluster"] == [D - FR.Z95 * se, D + FR.Z95 * se], R
        assert c["z_cluster"] == D / se, R
        s = stored["summary"]["interaction"]["by_regime"][R]
        assert s["D"] == D and s["cluster_se"] == se and s["z_cluster"] == D / se, R
        assert s["run_se"] == c["run_se"], R


def test_stored_summary_is_the_summary_of_the_stored_contrasts(stored):
    """Without rows: summary and outline_check are regime_summary and outline_check of
    the stored contrasts, as JSON values, so a changed contrast or summary number fails."""
    summ = FR.regime_summary(stored["contrasts"])
    js = (lambda x: json.loads(json.dumps(RS.clean(x))))
    assert js(summ) == stored["summary"]
    assert js(FR.outline_check(summ)) == stored["outline_check"]
    for name, w in FR.CONTRASTS.items():
        for R, c in stored["contrasts"].items():
            assert summ[name]["by_regime"][R]["D"] == c[name]["D"], (name, R)
            assert summ[name]["by_regime"][R]["cluster_se"] == c[name]["cluster_se"], (name, R)


def test_stored_contrasts_are_the_signed_sums_of_rs_differences(stored):
    """Without rows: every contrast's D is the signed sum of RS's stored, unrounded
    differences against ship (to float rounding), and the two-cell contrasts that RS's
    tables hold have RS's cluster SE (exactly, or to the tables' 6 decimals)."""
    with open(RS.OUT) as f:
        summary = json.load(f)["summary"]
    for R in RS.REGIMES:
        x = {X: summary["vs_ship"][R][X]["exact"] for X in ("eb_fit", "smcal", "smooth", "legacy")}
        d = {X: v["D"] for X, v in x.items()}
        d["ship"] = 0.0
        c = stored["contrasts"][R]
        for name, w in FR.CONTRASTS.items():
            want = sum(a * d[cfg] for cfg, a in w)
            assert math.isclose(c[name]["D"], want, rel_tol=0, abs_tol=1e-12), (R, name)
        for name, X, sign in FR.RS_EXACT:
            assert c[name]["cluster_se"] == x[X]["cluster_se"], (R, name)
            assert c[name]["D"] == sign * x[X]["D"], (R, name)
        for name, table, X, sign in FR.RS_TABLES:
            t = summary[table][R][X]["diff"]
            assert abs(c[name]["D"] - sign * t["mean"]) <= FR.TABLE_TOL, (R, name)
            assert abs(c[name]["cluster_se"] - t["cluster_se"]) <= FR.TABLE_TOL, (R, name)


def test_stored_file_recomputes_from_the_rows(stored):
    """With the rows: the script's blocks recomputed from data/regime_sensitivity_rows/
    equal the stored ones as JSON values (so the pins above are the rows' numbers)."""
    if not all(os.path.exists(RS.row_path(R, i)) for R in RS.REGIMES for i in range(RS.N_RUNS[R])):
        pytest.skip("data/regime_sensitivity_rows/ not present")
    res, checks = FR.compute(RS.load_rows(RS.ROWS))
    assert not checks.failed(), checks.failed()
    js = (lambda x: json.loads(json.dumps(RS.clean(x))))
    for k in ("design", "cells", "contrasts", "summary", "outline_check"):
        assert js(res[k]) == stored[k], k


def test_stored_interaction_headline(stored):
    """The outline's derived range at four decimals, now with paired cluster SEs."""
    s = stored["summary"]["interaction"]["testlike"]
    assert [round(x, 4) for x in s["D_range"]] == [-0.0155, -0.0125]
    assert s["all_negative"] and s["all_ci95_below_zero"]
    assert stored["outline_check"]["matches"]


def test_stored_file_carries_no_subject_id(stored):
    path = os.path.join(RS.ROWS, RS.COMPOSITIONS)
    if not os.path.exists(path):
        pytest.skip("data/regime_sensitivity_rows/ not present")
    with open(path) as f:
        comps = json.load(f)["runs"]
    subjects = {p["subject"] for runs in comps.values() for m in runs.values() for p in m}
    text = json.dumps(stored)
    assert subjects and not any(s in text for s in subjects)
    assert '"subject"' not in text
