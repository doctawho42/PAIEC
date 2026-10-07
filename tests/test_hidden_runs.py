"""experiments/hidden_runs.py: the identity it rests on, and its stored numbers.

Brier = ECE^2 + p(1-p) holds exactly for a pair that gets one prediction, with
ECE as paiec.official.ece computes it (response-weighted, ten bins) and p the
response-weighted rate; on a synthetic feedback table at the organisers' 6-dp
precision the script's identity reading finds such pairs constant across
budgets within its rounding bound and gives their p(1-p) as v, and a pair whose
predictions spread within the pair is not. The main numbers the script derives
from the committed tables (no data rows, no synthetic block) must equal those
stored in results/hidden_runs.json, the stored file must be this script's
output, carry no anonymous id and record every scratch value as matched or
explained.

The W2 additions of 2026-10-06: the pair SEs of B_k - v by budget recompute
from the tables; B31 - v equals ECE31^2 - R31 (0.0036 for hier, pooled or by
run) and is ECE31^2 alone for one prediction per pair; the distinct-pair
bootstrap merges exactly the one pair that is in run 1 and run 3; the stored
raw placement and the replica gap at B31 are consistent with the stored
replica numbers they derive from; the documents the hard-coded constants come
from are hashed, with the passage that holds them.

Added 2026-10-07: the three terms' pair SEs recompute from the tables, column
by column, in the recomputed and the stored blocks (so B31 counted in the
B1..B15 term fails); where data/regime_sensitivity_rows/ holds the rows the
stored file read, replica.placement_raw (z by budget), replica.late and
late_gap.vs_replica recompute from them by value (skipped otherwise); the
stored about.id_use states the id uses, and the distinct-pair block is
labelled a sensitivity analysis.
"""
import json
import math
import os
import re
import warnings

import numpy as np
import pytest

from experiments import formative_feedback as FF
from experiments import hidden_runs as HR
from paiec.official import ece

W = np.asarray(HR.WEIGHTS, float)


def _pair_scores(q, y):
    q = np.broadcast_to(np.asarray(q, float), y.shape)
    return float(np.mean((q - y) ** 2)), ece(q, y)


@pytest.mark.parametrize("q", [0.0, 0.03, 0.25, 0.5, 0.61, 0.9, 1.0])
def test_identity_is_exact_for_one_prediction(q):
    rng = np.random.default_rng(1)
    items = rng.random(57) < 0.37
    trials = rng.integers(1, 4, size=items.size)            # several responses per item
    y = np.repeat(items, trials).astype(float) * (rng.random(trials.sum()) < 0.9)
    o = y.mean()
    b, e = _pair_scores(q, y)
    assert math.isclose(e, abs(q - o), abs_tol=1e-12)
    assert math.isclose(b - e ** 2, o * (1 - o), abs_tol=1e-12)


def test_identity_fails_when_predictions_spread():
    rng = np.random.default_rng(2)
    y = (rng.random(80) < 0.4).astype(float)
    q = np.clip(0.4 + rng.normal(0, 0.25, y.size), 0.01, 0.99)
    b, e = _pair_scores(q, y)
    assert abs((b - e ** 2) - y.mean() * (1 - y.mean())) > 1e-3


def _synthetic_table(pairs):
    """The organisers' layout: an ALC summary and one section per budget."""
    out = ["Formative feedback", ""]
    rows = {}
    for j, (n, y, preds) in enumerate(pairs):
        sc = [_pair_scores(p, y) for p in preds]
        br = [round(b, 6) for b, _ in sc]
        ec = [round(e, 6) for _, e in sc]
        rows[j] = (f"subject_{100 + j}", f"benchmark_{200 + j % 2}", n, br, ec)
    out += ["ALC summary", "", "subject_id  Benchmark  N  Brier ALC  Calibration ECE", "---"]
    out += [f"{s}  {bm}  {n}  {W @ np.array(br):.6f}  {W @ np.array(ec):.6f}" for s, bm, n, br, ec in rows.values()]
    for k, B in enumerate(HR.BUDGETS):
        out += ["", f"Label budget: {B}", "", "subject_id  Benchmark  N  Brier score  Calibration ECE", "---"]
        out += [f"{s}  {bm}  {n}  {br[k]:.6f}  {ec[k]:.6f}" for s, bm, n, br, ec in rows.values()]
    return "\n".join(out) + "\n"


def test_identity_reading_on_a_synthetic_table(tmp_path):
    rng = np.random.default_rng(3)
    pairs, truth = [], []
    for j in range(4):
        n = int(rng.integers(44, 114))
        y = (rng.random(n) < rng.uniform(0.05, 0.8)).astype(float)
        preds = list(rng.uniform(0.02, 0.98, 6))             # one value per budget
        if j == 3:                                           # item-varying from budget 1 on
            preds = [preds[0]] + [np.clip(y * 0.6 + 0.2 + rng.normal(0, 0.1, n), 0.01, 0.99)
                                  for _ in range(5)]
        pairs.append((n, y, preds))
        truth.append(y.mean() * (1 - y.mean()))
    path = tmp_path / "run.txt"
    path.write_text(_synthetic_table(pairs))
    runs = HR.load_tables({1: str(path)})
    assert len(runs[1]) == 4
    ident = HR.identity(runs)["run1"]
    assert ident["pairs_within_rounding_bound"] == 3                # the item-varying pair is not
    rows = runs[1]
    for p, t in zip(rows, truth):                                    # v = B0 - ECE0^2 = p(1-p)
        assert abs(HR.pair_v(p) - t) <= 2e-6
    B = np.array([p["brier"] for p in rows[:3]])
    E = np.array([p["ece_b"] for p in rows[:3]])
    rg = (B - E ** 2).max(1) - (B - E ** 2).min(1)
    assert rg.max() <= 1e-6 * (1 + 2 * E.max()) + 1e-12
    with warnings.catch_warnings():       # the pooled bins of v are empty on three pairs
        warnings.simplefilter("ignore", RuntimeWarning)
        dec = HR.decomposition({1: rows[:3], 2: rows[:3], 3: rows[:3]})["run1"]
    assert math.isclose(dec["v"], float(np.mean(truth[:3])), abs_tol=2e-6)
    assert math.isclose(dec["alc_minus_v"], dec["b0_term"] + dec["b1_b15_terms"] + dec["b31_term"],
                        abs_tol=1e-12)
    # one prediction per pair: R31 is 0 up to the rounding, so B31 - v is ECE31^2 alone
    lg = HR.late_gap({1: rows[:3], 2: rows[:3], 3: rows[:3]})["run1"]
    assert abs(lg["r31"]) <= 2e-6 and abs(lg["identity_residual"]) < 1e-15
    assert math.isclose(lg["b31_minus_v"], lg["ece31_sq"], abs_tol=2e-6)
    # the item-varying pair has a within-pair resolution at B31 that a single value cannot have
    lg4 = HR.late_gap({1: rows, 2: rows, 3: rows})["run1"]
    assert lg4["r31"] * 4 > 1e-3 and abs(lg4["identity_residual"]) < 1e-15


# --- the stored file ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stored():
    if not os.path.exists(HR.OUT):
        pytest.skip("results/hidden_runs.json not written")
    with open(HR.OUT) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def recomputed():
    runs = HR.load_tables()
    res = HR.compute(runs, with_rows=False, with_requirement=False)
    return runs, res, HR.main_numbers(res)


def test_tables_are_the_stored_records():
    checks = HR.Checks()
    HR.check_tables(HR.load_tables(), checks)
    assert not checks.failed(), checks.failed()


def test_main_numbers_match_the_stored_file(stored, recomputed):
    _, _, main = recomputed
    want = stored["main"]
    for k, v in main.items():
        assert k in want, k
        if v is None:
            assert want[k] is None, k
        else:
            assert math.isclose(v, want[k], rel_tol=0, abs_tol=1e-12), (k, v, want[k])


def test_headline_values(recomputed):
    """The plan's section 2 and 5 numbers, at the precision it states them."""
    _, res, m = recomputed
    assert [round(m[f"v_run{r}"], 4) for r in (1, 2, 3)] == [0.1536, 0.1797, 0.1608]
    assert [round(m[f"alc_minus_v_run{r}"], 4) for r in (1, 2, 3)] == [0.0577, 0.0130, 0.0208]
    assert round(m["legacy_minus_hier_vadj"], 4) == 0.0405 and round(m["legacy_minus_hier_vadj_se"], 3) == 0.019
    assert round(m["held_out_tl_shipped_minus_legacy"], 4) == -0.0419
    assert round(m["held_out_tl_cluster_se"], 4) == 0.0037
    assert round(m["v_run_strat_mean"], 3) == 0.160 and round(m["v_run_strat_sd"], 3) == 0.017
    assert round(m["ideal_lower_team_hier"], 3) == 0.016 and round(m["ideal_prior_weighted_hi"], 3) == 0.032
    assert res["identity"]["run1"]["range_of_run_mean_over_budgets"] < 1e-6
    assert res["identity"]["run1"]["pairs_within_rounding_bound"] == 9
    assert round(m["curve_hidden_b7_b31_mean"], 4) == -0.0012


def test_stored_file_is_this_scripts_output(stored):
    assert stored["provenance"]["script_sha256"] == HR.sha256(os.path.abspath(HR.__file__))
    assert stored["scratch"]["counts"]["differs"] == 0
    assert all(c["ok"] for c in stored["checks"] if c["required"])
    for rel, digest in stored["provenance"]["inputs_sha256"].items():
        p = os.path.join(HR.ROOT, rel)
        if os.path.isfile(p) and rel.startswith("results/"):
            assert HR.sha256(p) == digest, rel


def test_stored_file_carries_no_anonymous_id(stored, recomputed):
    runs, _, _ = recomputed
    text = json.dumps(stored)
    ids = {p[k] for P in runs.values() for p in P for k in ("subject", "benchmark")}
    assert ids and not any(i in text for i in ids)
    assert not re.search(r"subject_\d|benchmark_\d", text)


def test_pair_ses_recompute_from_the_tables(recomputed):
    runs, res, m = recomputed
    dec = res["decomposition"]
    for name, P in (("run1", runs[1]), ("run2", runs[2]), ("run3", runs[3]),
                    ("hier_pooled", runs[2] + runs[3])):
        B, E, v = HR.arrays(P)
        ex = B - v[:, None]
        se = dec[name]["pair_se"]
        assert np.allclose(se["excess_by_budget_pair_se"], ex.std(0, ddof=1) / np.sqrt(len(P)), rtol=0, atol=1e-15)
        assert math.isclose(se["alc_minus_v_pair_se"], (B @ W - v).std(ddof=1) / np.sqrt(len(P)), abs_tol=1e-15)
        assert math.isclose(se["terms_pair_se"]["b31_term"], 0.1 * se["excess_by_budget_pair_se"][5], abs_tol=1e-15)
    for r in (1, 2, 3):                                # the same SE as vadjusted's, from the other block
        assert math.isclose(dec[f"run{r}"]["pair_se"]["alc_minus_v_pair_se"], res["vadjusted"][f"run{r}"]["se"],
                            abs_tol=1e-15)
    assert dec["hier_pooled"]["pairs"] == 17
    assert round(m["hier_pooled_alc_minus_v_se"], 4) == 0.0027
    assert round(m["excess_se_B31_hier_pooled"], 4) == 0.0024


def _terms_pair_se(P):
    """The three weighted terms' pair SEs, written out column by column (B1, B3, B7 and B15
    in the middle term; B31 alone in the last), independently of pair_spread's slicing."""
    B, _, v = HR.arrays(P)
    ex = B - v[:, None]
    terms = {"b0_term": 0.1 * ex[:, 0],
             "b1_b15_terms": 0.2 * ex[:, 1] + 0.2 * ex[:, 2] + 0.2 * ex[:, 3] + 0.2 * ex[:, 4],
             "b31_term": 0.1 * ex[:, 5]}
    return ({k: float(np.std(t, ddof=1) / math.sqrt(len(P))) for k, t in terms.items()},
            {k: float(np.mean(t)) for k, t in terms.items()})


def _pair_groups(runs):
    return (("run1", runs[1]), ("run2", runs[2]), ("run3", runs[3]), ("hier_pooled", runs[2] + runs[3]))


def test_terms_pair_se_recompute_from_the_tables(stored, recomputed):
    """Each of the three terms' pair SEs (and the run blocks' term means) from the tables,
    in the recomputed blocks and in the stored file: adding B31 to the B1..B15 term, or
    dropping a budget from it, moves b1_b15_terms well past the tolerance."""
    runs, res, _ = recomputed
    for name, P in _pair_groups(runs):
        se, mean = _terms_pair_se(P)
        for blocks in (res["decomposition"], stored["decomposition"]):
            got = blocks[name]["pair_se"]["terms_pair_se"]
            assert set(got) == set(se), name
            for k in se:
                assert math.isclose(got[k], se[k], rel_tol=0, abs_tol=1e-15), (name, k, got[k], se[k])
            if name != "hier_pooled":                       # the pooled block stores no term means
                for k in mean:
                    assert math.isclose(blocks[name][k], mean[k], rel_tol=0, abs_tol=1e-15), (name, k)
    # the mutant this pins: B31 counted in the B1..B15 term changes its SE in every group
    for name, P in _pair_groups(runs):
        B, _, v = HR.arrays(P)
        ex = B - v[:, None]
        wrong = float((0.2 * ex[:, 1:6]).sum(1).std(ddof=1) / math.sqrt(len(P)))
        assert abs(wrong - stored["decomposition"][name]["pair_se"]["terms_pair_se"]["b1_b15_terms"]) > 1e-6, name


def test_late_gap_is_ece2_minus_resolution(recomputed):
    """B31 - v = ECE31^2 - R31 in the B0 convention, and its value 0.0036 for hier."""
    _, res, _ = recomputed
    lg, dec, sf = res["late_gap"], res["decomposition"], res["split_floor"]
    for k in ("run1", "run2", "run3", "hier_pooled"):
        assert abs(lg[k]["b31_minus_v"] - (lg[k]["ece31_sq"] - lg[k]["r31"])) < 1e-15
        assert math.isclose(lg[k]["r31"], dec[k]["within_pair_resolution_b31"]["mean"], abs_tol=1e-15) \
            or k == "run1"
    for k in ("run2", "run3"):
        assert math.isclose(lg[k]["ece31_sq"], sf[k]["platform_excess_B31"], abs_tol=1e-15)
        assert math.isclose(lg[k]["b31_minus_v"], dec[k]["b31_minus_v"], abs_tol=1e-15)
        # the floor with v from B0 sits above split_floor's (v = B31 - ECE31^2) by v's R31 share
        assert 0 < lg[k]["floor_pure_rate_learner_B31_b0_convention"] - sf[k]["floor_pure_rate_learner_B31"] < 3e-4
    assert lg["value_4dp"] == {"run1": 0.0031, "run2": 0.0036, "run3": 0.0036, "hier_pooled": 0.0036}
    assert abs(lg["run1"]["r31"]) < 1e-6                  # LegacyP: one prediction per pair
    assert lg["hier_pooled"]["pairs_gt_0.03"] == 0


def test_distinct_pairs_merge_only_the_repeated_pair(stored, recomputed):
    runs, res, _ = recomputed
    dp = res["run_spread"]["distinct_pairs"]
    keys = [(p["subject"], p["benchmark"]) for r in (1, 2, 3) for p in runs[r]]
    assert dp["appearances"] == len(keys) == 26 and dp["distinct_pairs"] == len(set(keys)) == 25
    assert dp["pairs_in_two_runs"] == 1 and dp["pairs_in_three_runs"] == 0 and dp["repeat_runs"] == [1, 3]
    assert dp["repeat_same_n"] and dp["repeat_max_abs_v_difference"] <= 2e-6
    assert not dp["repeat_in_one_run"] and not dp["repeat_within_hier_runs"]
    hier = [(p["subject"], p["benchmark"]) for r in HR.HIER for p in runs[r]]
    assert len(set(hier)) == len(hier)                    # hier's ALC bootstrap is not touched
    # the block's per-appearance version is the stored seed-7 one up to Monte Carlo noise
    for kind in ("v_iid", "v_strat"):
        s7 = stored["run_spread"][kind]
        s8 = dp[kind]["per_appearance"]
        p = s7["P_le_0.117"]
        mc = math.sqrt(p * (1 - p) * (1 / HR.NBOOT + 1 / HR.NBOOT_DISTINCT))
        assert abs(s8["P_le_0.117"] - p) <= 4 * mc, kind
        assert abs(s8["mean"] - s7["mean"]) <= 4 * s7["sd"] * math.sqrt(1 / HR.NBOOT + 1 / HR.NBOOT_DISTINCT), kind
        d = dp[kind]["distinct_minus_per_appearance"]
        assert math.isclose(d["P_le_0.117"], dp[kind]["distinct"]["P_le_0.117"] - s8["P_le_0.117"], abs_tol=1e-12)


def test_id_use_and_the_sensitivity_are_stated(stored):
    """about.id_use names the two analytic uses (the repeated pair, the two stratified
    bootstraps) and the bookkeeping ones; the distinct-pair block is the sensitivity."""
    use = stored["about"]["id_use"]
    for s in ("find the one subject-benchmark pair that appears in two runs", "two stratified bootstraps",
              "run_spread.v_strat and alc_hier_strat", "run_spread.distinct_pairs.v_strat", "check_tables",
              "No id is written to this file"):
        assert s in use, s
    dp = stored["run_spread"]["distinct_pairs"]
    assert dp["role"].startswith("sensitivity analysis") and dp["note"].startswith("A sensitivity analysis.")
    assert "primary" in stored["about"]["primary_and_sensitivity"]


def test_doc_passage_finds_heading_paragraphs_and_anchored_lines(tmp_path, monkeypatch):
    text = "# T\n\n### Against the live leaderboard\n\nRead on 2026-09-24: entry\n0.1801 and best 0.1172.\n\nNext.\n" \
           "*Figure A1.* The line $0.025 + 1.2118\\,p$.\n"
    p = tmp_path / "doc.md"
    p.write_text(text)
    lines = text.split("\n")
    assert HR.doc_passage(lines, {"heading": "### Against the live leaderboard"}) == (5, 6)
    assert HR.doc_passage(lines, {"line_starts": "*Figure A1.*"}) == (9, 9)
    specs = {"lb": {"path": str(p), "heading": "### Against the live leaderboard",
                    "values": {"a": "0.1801", "b": "0.1172", "c": "2026-09-24"}},
             "line": {"path": str(p), "line_starts": "*Figure A1.*", "values": {"l": "0.025 + 1.2118"}},
             "moved": {"path": str(p), "heading": "### Gone", "values": {"a": "0.1801"}},
             "changed": {"path": str(p), "line_starts": "*Figure A1.*", "values": {"l": "0.025 + 1.3"}}}
    monkeypatch.setattr(HR, "DOC_CONSTANTS", specs)
    got = HR.doc_constants()
    assert got["lb"]["found"] and got["line"]["found"]            # wrapped lines are whitespace-normalised
    assert not got["moved"]["found"] and got["moved"]["lines"] is None
    assert not got["changed"]["found"]


def test_doc_constants_are_hashed_with_their_passage(stored):
    pv = stored["provenance"]
    dc = pv["doc_constants"]
    assert set(dc) == set(HR.DOC_CONSTANTS)
    assert dc["leaderboard"]["values"] == {"organisers_entry": str(HR.LEADERBOARD["organisers_entry"]),
                                           "best_entry": str(HR.LEADERBOARD["best_entry"]),
                                           "read": HR.LEADERBOARD["read"]}
    assert dc["empirical_mean_line"]["values"] == {"line": f"{HR.EMP_LINE[0]} + {HR.EMP_LINE[1]}"}
    current = HR.doc_constants()
    for name, rec in dc.items():
        assert rec["found"], name
        assert re.fullmatch(r"[0-9a-f]{64}", rec["sha256"]) and re.fullmatch(r"[0-9a-f]{64}", rec["passage_sha256"])
        assert pv["inputs_sha256"][rec["path"]] == rec["sha256"]
        if current[name]["sha256"] == rec["sha256"]:     # the document unchanged since the run: same passage
            assert current[name] == rec, name


def test_stored_raw_placement_and_replica_gap_are_consistent(stored):
    rp = stored["replica"]
    if rp is None:
        pytest.skip("replica rows were not read")
    alc = {"run1 (LegacyP)": stored["tables"]["run1"]["alc"], "run2 (hier)": stored["tables"]["run2"]["alc"],
           "run3 (hier)": stored["tables"]["run3"]["alc"],
           "runs 2+3 (hier)": (8 * stored["tables"]["run2"]["alc"] + 9 * stored["tables"]["run3"]["alc"]) / 17}
    for reg in HR.RS_REGIMES:
        for name, x in rp["placement_raw"][reg].items():
            nz = rp["noise"][reg][x["arm"]]
            assert x["arm"] == rp["placement_z"][reg][name]["arm"]
            assert math.isclose(x["hidden_alc"], alc[name], abs_tol=1e-12)
            assert math.isclose(x["regime_alc"], nz["alc"], abs_tol=1e-12)
            assert math.isclose(x["regime_alc_sd"], nz["alc_sd"], abs_tol=1e-12)
            assert math.isclose(x["z_alc"], (x["hidden_alc"] - nz["alc"]) / nz["alc_sd"], abs_tol=1e-9)
            sbs = rp["placement_raw_summary"]["side_by_side_z_alc"][name][reg]
            assert sbs == {"raw": x["z_alc"], "v_adjusted": rp["placement_z"][reg][name]["z_alc_minus_v"]}
        assert math.isclose(rp["late"][reg]["ship"]["b31_minus_v"], rp["validation"][reg]["b31_minus_v"],
                            abs_tol=1e-12)
        lg = stored["late_gap"]["vs_replica"][reg]
        for name, arm in (("hier_pooled", "ship"), ("run1", "legacy")):
            want = stored["late_gap"][name]["b31_minus_v"] - rp["late"][reg][arm]["b31_minus_v_b0_convention"]
            assert lg[name]["arm"] == arm and math.isclose(lg[name]["hidden_minus_replica"], want, abs_tol=1e-15)


@pytest.fixture(scope="module")
def replica_recomputed(stored, recomputed):
    """replica and late_gap recomputed from data/regime_sensitivity_rows, if those rows are
    on disk and are the ones the stored file read (its inputs digest); else skip."""
    if stored["replica"] is None:
        pytest.skip("the stored file read no replica rows")
    want = stored["provenance"]["inputs_sha256"]
    for reg in HR.RS_REGIMES:
        p = os.path.join(HR.RS_ROWS, reg)
        if not os.path.isdir(p):
            pytest.skip("data/regime_sensitivity_rows/ not present")
        if HR.dir_digest(p) != want.get(HR.rel(p) + "/*.json"):
            pytest.skip(f"the {reg} rows on disk are not the ones the stored file read")
    runs = recomputed[0]
    rep = HR.replica(runs, HR.RS_ROWS)
    return rep, HR.late_gap(runs, rep)


def _close(a, b, tol=1e-12):
    return a is b or (a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=tol))


def test_replica_placement_raw_late_and_gap_recompute_from_the_rows(stored, replica_recomputed):
    """By value, from the rows: replica.placement_raw (z by budget, z of ALC and the
    numbers they divide), replica.late, and late_gap.vs_replica with its summary."""
    rep, lg = replica_recomputed
    srep, slg = stored["replica"], stored["late_gap"]
    assert set(srep["placement_raw"]) == set(rep["placement_raw"]) == set(HR.RS_REGIMES)
    for reg in HR.RS_REGIMES:
        assert set(srep["placement_raw"][reg]) == set(rep["placement_raw"][reg])
        for name, x in rep["placement_raw"][reg].items():
            s = srep["placement_raw"][reg][name]
            assert s["arm"] == x["arm"]
            assert len(s["z_by_budget"]) == len(x["z_by_budget"]) == len(HR.BUDGETS)
            assert all(_close(a, b) for a, b in zip(s["z_by_budget"], x["z_by_budget"])), (reg, name)
            for k in ("hidden_alc", "regime_alc", "regime_alc_sd", "z_alc"):
                assert _close(s[k], x[k]), (reg, name, k)
        assert set(srep["late"][reg]) == set(rep["late"][reg]) == {"ship", "legacy"}
        for arm, x in rep["late"][reg].items():
            s = srep["late"][reg][arm]
            assert set(s) == set(x)
            assert s["runs"] == x["runs"] and s["appearances"] == x["appearances"], (reg, arm)
            for k in ("b31_minus_v", "b31_minus_v_run_se", "b31_minus_v_b0_convention",
                      "b31_minus_v_b0_convention_run_se"):
                assert _close(s[k], x[k]), (reg, arm, k)
        assert set(slg["vs_replica"][reg]) == set(lg["vs_replica"][reg]) == {"hier_pooled", "run1"}
        for name, x in lg["vs_replica"][reg].items():
            s = slg["vs_replica"][reg][name]
            assert s["arm"] == x["arm"]
            for k in ("replica_b31_minus_v_b0_convention", "hidden_minus_replica", "se"):
                assert _close(s[k], x[k]), (reg, name, k)
    for name, x in lg["vs_replica_summary"].items():
        for k, v in x.items():
            assert all(_close(a, b) for a, b in zip(slg["vs_replica_summary"][name][k], v)), (name, k)
    for name, v in rep["placement_raw_summary"]["z_alc_range_test_like"].items():
        assert all(_close(a, b) for a, b in zip(srep["placement_raw_summary"]["z_alc_range_test_like"][name], v))


def test_parser_is_the_record_stage():
    """The port reads the tables through formative_feedback.parse, nothing else."""
    for r, path in HR.TABLES.items():
        with open(path) as f:
            assert HR.load_tables({r: path})[r] == FF.parse(f.read())
