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


def test_parser_is_the_record_stage():
    """The port reads the tables through formative_feedback.parse, nothing else."""
    for r, path in HR.TABLES.items():
        with open(path) as f:
            assert HR.load_tables({r: path})[r] == FF.parse(f.read())
