"""experiments/e1_baselines_rs.py (E1, plan P2 amendment 2).

The plan fills exactly the regimes baselines_p1 left out, with baselines_p1's own two
configs, seed, runs and checks; the E1 declaration and the scoring code are pinned;
the statistics are baselines_p1's on merged rows (synthetic data). Where the stored
rows are on disk, the script reproduces baselines_p1's TUNED numbers for EmpMean and
1PL-ship bit for bit: its statistics on baselines_p1's rows give the stored results
exactly, and, with the measurement-db download, re-scoring TUNED runs gives the stored
rows exactly (empmean on all 80 runs, the 1PL on two). The stored result, if written,
records the same for all 80 runs, and reads its rows from the repository's gitignored
default directory. summarise refuses (non-zero exit, nothing written) when a planned row
is missing."""
import copy
import hashlib
import json
import math
import os

import numpy as np
import pytest

from experiments import baselines_p1 as BP
from experiments import e1_baselines_rs as E
from experiments import regime_sensitivity as RS

W6 = np.asarray(BP.WEIGHTS, float)
DATA = os.environ.get("PAIEC_DATA", os.path.join(E.ROOT, "data"))


def have_rows(R, i=None):
    ids = range(E.N_RUNS[R]) if i is None else (i,)
    return all(os.path.exists(RS.row_path(R, j)) and os.path.exists(BP.row_path(R, j)) for j in ids)


def have_download():
    return all(os.path.exists(os.path.join(DATA, b, "response.parquet"))
               for b in ("matharena", "multi_swebench", "real_webagents", "researchcodebench",
                         "swe_rebench"))


needs_rows = pytest.mark.skipif(not (have_rows("TUNED") and os.path.exists(BP.OUT)),
                                reason="P1a's and baselines_p1's TUNED rows are not on disk")
needs_download = pytest.mark.skipif(not have_download(), reason="no local measurement-db download")


# --- the plan --------------------------------------------------------------------------------

def test_the_plan_fills_the_regimes_baselines_p1_left():
    assert E.REGIMES == ("READING", "AUDIT", "MIXTURE", "FLAT")
    assert set(E.REGIMES).isdisjoint(BP.REGIMES) and set(E.REGIMES) | set(BP.REGIMES) == set(RS.REGIMES)
    assert all(RS.REGIMES[R][0] == "testlike" for R in E.REGIMES)
    assert E.CONTROL == ("TUNED",) and set(E.CONTROL) <= set(BP.REGIMES)
    assert E.SEED == BP.SEED == RS.SCORE_SEED == 11 and E.SPLIT_SCOPE == BP.SPLIT_SCOPE == "pair"
    assert E.N_RUNS == {R: 80 for R in E.SCORED} and all(E.N_RUNS[R] == RS.N_RUNS[R] for R in E.SCORED)
    # the two comparators are baselines_p1's, the very objects, and nothing else is scored
    assert E.NEW == ("empmean", "rasch") and set(E.NEW) <= set(BP.NEW)
    assert all(E.CONFIGS[c] is BP.CONFIGS[c] for c in E.NEW)
    assert BP.CONFIGS["rasch"]["level"] == "LEVEL" and BP.CONFIGS["empmean"]["kind"] == "empirical"
    assert E.STORED == BP.STORED and E.SMCAL == BP.SMCAL and E.REPRO_TOL == BP.REPRO_TOL
    assert (E.CHECK_EVERY, E.CHECK_SOME) == (BP.CHECK_EVERY, BP.CHECK_SOME)
    assert E.CHECK_RUNS == {R: (0, 79) for R in E.SCORED}
    tasks = E.plan_tasks()
    assert len(tasks) == len(set(tasks)) == 400
    assert tasks[:5] == [(R, 0) for R in E.SCORED]
    shards = [E.shard_tasks(k, 3) for k in range(3)]
    assert sorted(t for s in shards for t in s) == sorted(tasks)
    assert set(E.T3_COLUMNS) == set(E.SCORED) | set(BP.REGIMES)


def test_the_e1_declaration_and_the_scoring_code_are_pinned():
    with open(os.path.join(E.ROOT, E.PLAN_PATH), encoding="utf-8") as f:
        plan = f.read()
    bullet = E.e1_bullet(plan)
    assert bullet.startswith(E.E1_BULLET_START) and bullet.endswith("selects anything.\n")
    assert hashlib.sha256(bullet.encode("utf-8")).hexdigest() == E.E1_BULLET_SHA256
    assert "READING, AUDIT, MIXTURE, FLAT" in " ".join(bullet.split())
    assert E.e1_bullet("no such bullet\n") is None
    assert E.e1_bullet("x\n" + E.E1_BULLET_START + " y\n  z\n- next\n") == E.E1_BULLET_START + " y\n  z\n"
    block = E.scoring_block()
    assert block.startswith(E.PLAN_MARK) and hashlib.sha256(block.encode()).hexdigest() == E.SCORING_SHA
    for name in ("def score_task", "def stage_score", "NEW = ", "EXACT_FIELDS = ", "def exact_diff"):
        assert name in block
    assert "def regime_summary" not in block and E.STATS_MARK not in block


@pytest.mark.skipif(E.git("rev-parse", E.AMENDMENT_COMMIT) is None, reason="amendment commit not available")
def test_the_lock_holds_in_this_checkout():
    res = E.lock_checks()
    assert res == {k: True for k in res}, res


# --- bit-for-bit comparison and statistics ---------------------------------------------------

def test_exact_diff_sees_one_ulp():
    a = {"b": [[0.1, 0.2, 0.3, 0.25, 0.2, 0.15]], "ece": [[0.0] * 6], "q0": [0.4], "calls": 12,
         "failures": 0, "unconverged": 0, "secs": 1.0}
    b = copy.deepcopy(a)
    b["secs"] = 9.0                                         # timings are not compared
    assert E.exact_diff(a, b) == ([], 0.0)
    b["b"][0][3] = math.nextafter(0.25, 1.0)
    bad, d = E.exact_diff(a, b)
    assert bad == ["b"] and 0 < d < 1e-15
    b = copy.deepcopy(a)
    b["calls"] = 13
    assert E.exact_diff(a, b)[0] == ["calls"]


#: synthetic Brier offsets per config
OFFSET = {"legacy": 0.04, "smooth": 0.02, "smcal": 0.006, "eb_fit": 0.025, "onepl": 0.002,
          "rasch": 0.004, "rasch_fit": 0.012, "empmean": 0.05}


def synthetic_rows(runs=14, seed=0, configs=E.NEW):
    """P1a-like rows (every RS config) and rows of `configs`, on the same runs."""
    rng = np.random.default_rng(seed)
    old, new = {}, {}
    for i in range(runs):
        k = int(rng.integers(3, 8))
        meta = [{"bench": f"{q}::g{j}", "parent": q, "subject": f"s{int(rng.integers(0, 5))}",
                 "eval": 50, "p": 0.3, "logit": float(rng.normal(-1, 1.5))}
                for j, q in enumerate(rng.choice(list(RS.PARENTS), k))]
        base = rng.uniform(0.05, 0.3, (k, 6))

        def res(c):
            b = np.clip(base + OFFSET.get(c, 0.0) + rng.normal(0, 0.01, (k, 6)), 0, 1)
            return {"b": b.tolist(), "ece": (b / 2).tolist(), "q0": list(rng.random(k)), "calls": 10,
                    "mean_s": 0.001, "max_s": 0.01, "failures": 0, "unconverged": 0, "secs": 0.1}
        old[i] = {"meta": meta, "res": {c: res(c) for c in RS.CONFIGS}}
        new[i] = {"meta": json.loads(json.dumps(meta)), "res": {c: res(c) for c in configs}}
    return old, new


def test_the_statistics_are_baselines_p1s():
    """On rows holding every baselines_p1 config, E1's summary of its two configs equals
    baselines_p1's part() on baselines_p1's own merge, as JSON values; D is the mean over
    runs of the run's pair-mean ALC difference and its cluster SE the bootstrap's sd."""
    old, new = synthetic_rows(configs=BP.NEW)
    summ, G = E.regime_summary("READING", old, new)
    H = BP.Rows("READING", BP.merge(old, new))
    js = (lambda x: json.loads(json.dumps(RS.clean(x))))
    for c in E.NEW:
        mine = {k: v for k, v in js(summ["vs_ship"][c]).items() if k != "U95"}
        assert mine == js(BP.part(H, c, "ship"))
        assert js(summ["vs_legacy"][c]) == js(BP.part(H, c, "legacy"))
        run_d = [np.mean(np.asarray(new[i]["res"][c]["b"]) @ W6 - np.asarray(old[i]["res"]["ship"]["b"]) @ W6)
                 for i in sorted(old)]
        assert summ["vs_ship"][c]["D"] == pytest.approx(np.mean(run_d), abs=1e-12)
        assert summ["vs_ship"][c]["cluster_se"] == pytest.approx(
            np.std(G.boot_dist(c, "ship"), ddof=1), abs=1e-12)
        assert summ["vs_ship"][c]["U95"] == pytest.approx(
            summ["vs_ship"][c]["D"] + 1.96 * summ["vs_ship"][c]["cluster_se"])
        assert summ["headline"][c]["cell"] == f"{summ['vs_ship'][c]['D']:+.4f} ({summ['vs_ship'][c]['cluster_se']:.4f})"
    assert summ["vs_ship"]["empmean"]["D"] > summ["vs_ship"]["rasch"]["D"] > 0
    # the agreement check reads its own output as equal, and sees a one-ulp change
    stored = {"vs_ship": {c: BP.part(H, c, "ship") for c in E.NEW},
              "vs_legacy": {c: BP.part(H, c, "legacy") for c in E.NEW},
              "configs": {c: {**H.config_summary(c), "mean_call_ms": 0.0, "max_call_s": 0.0} for c in E.NEW}}
    stored = js(stored)
    assert all(v["equal"] for v in E.bp_agreement(summ, stored).values())
    stored["vs_ship"]["rasch"]["cluster_se"] = math.nextafter(stored["vs_ship"]["rasch"]["cluster_se"], 1.0)
    got = E.bp_agreement(summ, stored)
    assert got["vs_ship.rasch"] == {"equal": False, "keys_differing": ["cluster_se"]}
    assert all(v["equal"] for k, v in got.items() if k != "vs_ship.rasch")


def test_merge_refuses_rows_that_do_not_pair():
    old, new = synthetic_rows(runs=4)
    bad = copy.deepcopy(new)
    bad[2]["meta"][0]["subject"] = "someone else"
    with pytest.raises(ValueError):
        E.merge(old, bad)
    bad = copy.deepcopy(new)
    del bad[1]["res"]["rasch"]
    with pytest.raises(ValueError):
        E.merge(old, bad)
    with pytest.raises(ValueError):
        E.merge(old, {i: new[i] for i in (0, 1, 2)})
    assert set(E.merge(old, new)[0]["res"]) == set(RS.CONFIGS) | set(E.NEW)


def test_the_scored_digest_ignores_timings_only():
    _, new = synthetic_rows(runs=3)
    d = E.scored_digest(new)
    other = copy.deepcopy(new)
    other[1]["res"]["rasch"]["secs"] = 99.0
    other[1]["res"]["rasch"]["mean_s"] = 9.0
    assert E.scored_digest(other) == d
    other[1]["res"]["rasch"]["b"][0][0] = math.nextafter(other[1]["res"]["rasch"]["b"][0][0], 1.0)
    assert E.scored_digest(other) != d


def test_the_t3_table_takes_public_columns_from_baselines_p1():
    old, new = synthetic_rows()
    summ, _ = E.regime_summary("READING", old, new)
    bp = {"regimes": {R: {"vs_ship": {c: {"D": 0.01 * (j + 1), "cluster_se": 0.001} for c in E.NEW},
                          "realised": {"mean": -0.5}} for j, R in enumerate(BP.REGIMES)}}
    t = E.t3_table({"READING": summ}, bp)
    for c in E.NEW:
        cols = t[c]["columns"]
        assert list(cols) == ["TUNED", "READING", "R1B", "R1P"]
        assert cols["READING"]["source"].startswith("E1 ") and cols["READING"]["D"] == summ["vs_ship"][c]["D"]
        assert cols["R1B"]["source"] == f"BP summary.regimes.R1B.vs_ship.{c}" and cols["R1B"]["cell"] == "+0.0200 (0.0010)"
        assert cols["TUNED"]["source"].startswith("BP ")      # no control scored here: read from BP


# --- baselines_p1's TUNED numbers, bit for bit --------------------------------------------

@needs_rows
def test_the_statistics_reproduce_baselines_p1s_tuned_numbers_bit_for_bit():
    """E1's statistics on P1a's TUNED rows and baselines_p1's stored TUNED rows give
    baselines_p1's stored vs_ship, vs_legacy and config numbers for empmean and the
    1PL exactly (every key, as JSON values; timings aside)."""
    old, new = {}, {}
    for i in range(E.N_RUNS["TUNED"]):
        old[i] = BP.stored_row("TUNED", i)
        with open(BP.row_path("TUNED", i)) as f:
            new[i] = json.load(f)
    summ, _ = E.regime_summary("TUNED", old, new)
    with open(BP.OUT) as f:
        stored = json.load(f)["summary"]["regimes"]["TUNED"]
    got = E.bp_agreement(summ, stored)
    assert got == {k: {"equal": True, "keys_differing": []} for k in got}, got
    assert summ["vs_ship"]["empmean"]["D"] == stored["vs_ship"]["empmean"]["D"]
    assert summ["vs_ship"]["rasch"]["cluster_se"] == stored["vs_ship"]["rasch"]["cluster_se"]


def few_parent_runs(n=2):
    """TUNED runs with the fewest parents (the cheapest to re-score), avoiding the
    check runs where ship, onepl and legacy are re-scored too."""
    with open(os.path.join(BP.RS_ROWS, RS.COMPOSITIONS)) as f:
        comps = json.load(f)["runs"]["TUNED"]
    key = sorted((len({p["parent"] for p in m}), sum(p["eval"] for p in m), int(i)) for i, m in comps.items())
    return [i for _, _, i in key if i not in E.CHECK_RUNS["TUNED"]][:n]


@needs_rows
@needs_download
def test_rescoring_reproduces_baselines_p1s_tuned_rows_bit_for_bit():
    """E1's own scoring path (score_task: the draw, the composition checks, the
    checkpoints, baselines_p1's factory) on two TUNED runs gives baselines_p1's stored
    rows for both configs exactly and passes every reproduction check; empmean, which
    costs no prior fit, is re-scored on all 80 TUNED runs."""
    _, comps = E.p1a_state()
    lib = BP.lib_digests()
    if BP.stored_row("TUNED", 0)["lib"] != lib:
        pytest.skip("the library differs from the one the stored rows record")
    # The stored rows were scored on one BLAS thread. Inside a pytest session numpy
    # is already loaded with the default thread count, so the environment variables
    # baselines_p1 sets at import come too late; limit the pools at run time instead.
    threadpoolctl = pytest.importorskip("threadpoolctl")
    with threadpoolctl.threadpool_limits(1):
        for i in few_parent_runs():
            row = E.score_task("TUNED", i, comps, lib)
            assert {f"baselines_p1:{c}" for c in E.NEW} <= set(row["checks"])
            assert all(c["ok"] for c in row["checks"].values()), row["checks"]
            assert row["checks"]["baselines_p1:rasch"]["max_abs_brier"] == 0.0
    import warnings
    from experiments import level_calibration as LC
    for i in range(E.N_RUNS["TUNED"]):
        with open(BP.row_path("TUNED", i)) as f:
            want = json.load(f)["res"]["empmean"]
        key, run = RS.draw("TUNED", E.SEED, i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slots, cps = LC.checkpoints(run, scope=E.SPLIT_SCOPE)
        got = BP.score_one("empmean", BP.Maker("empmean", run), slots, cps)
        assert E.exact_diff(got, want) == ([], 0.0), i


# --- the stored result ----------------------------------------------------------------------

def stored_result():
    if not os.path.exists(E.OUT):
        pytest.skip("results/e1_baselines_rs.json not written")
    with open(E.OUT) as f:
        return json.load(f)


def test_the_stored_result_holds_its_checks_and_the_control():
    res = stored_result()
    assert res["all_checks_hold"], [k for k, v in res["checks"].items() if not v]
    s = res["summary"]
    assert s["complete"] and s["scored"] == {R: 80 for R in E.SCORED}
    ctl = s["regimes"]["TUNED"]["control"]
    assert ctl["rows_equal_to_baselines_p1"] == {c: "80 of 80" for c in E.NEW}
    assert ctl["max_abs_brier_vs_baselines_p1"] == {c: 0.0 for c in E.NEW}
    assert all(v["equal"] for v in ctl["statistics_equal_to_baselines_p1"].values())
    assert res["provenance"]["scoring_block_sha256"] == E.SCORING_SHA
    for R in E.SCORED:
        assert res["rows"]["regimes"][R]["scoring_sha256"] == [E.SCORING_SHA]
    # the rows the result read are in the repository's (gitignored) default directory
    assert res["rows"]["dir"] == res["rows"]["default_dir"] == os.path.relpath(E.ROWS, E.ROOT)
    assert not os.path.isabs(res["rows"]["dir"])


def test_summarise_refuses_missing_rows(tmp_path):
    """With no rows (an empty directory, or one regime short) summarise exits non-zero and
    writes nothing; --allow-missing is the only way to a partial summary."""
    rows = tmp_path / "rows"
    rows.mkdir()
    out = tmp_path / "e1.json"
    with pytest.raises(SystemExit) as e:
        E.main(["summarise", "--rows", str(rows), "--out", str(out)])
    assert e.value.code not in (0, None) and "missing" in str(e.value.code)
    assert not out.exists()
    assert E.missing_rows(str(rows)) == {R: list(range(E.N_RUNS[R])) for R in E.SCORED}
    # every row but one present (empty files: only their existence is checked first)
    for R, i in E.plan_tasks():
        p = E.row_path(R, i, rows=str(rows))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if (R, i) != ("FLAT", 79):
            open(p, "w").close()
    assert E.missing_rows(str(rows)) == {"FLAT": [79]}
    with pytest.raises(SystemExit) as e:
        E.main(["summarise", "--rows", str(rows), "--out", str(out)])
    assert e.value.code not in (0, None) and not out.exists()


def test_the_stored_t3_cells_read_their_numbers():
    res = stored_result()
    with open(BP.OUT) as f:
        bp = json.load(f)["summary"]["regimes"]
    for c in E.NEW:
        cols = res["summary"]["t3"][c]["columns"]
        assert list(cols) == list(E.T3_COLUMNS)
        for R, v in cols.items():
            src = res["summary"]["regimes"][R] if R in E.SCORED else bp[R]
            assert v["D"] == src["vs_ship"][c]["D"] and v["cluster_se"] == src["vs_ship"][c]["cluster_se"]
            assert v["cell"] == f"{v['D']:+.4f} ({v['cluster_se']:.4f})"
        assert cols["TUNED"]["D"] == bp["TUNED"]["vs_ship"][c]["D"]


def test_the_stored_result_recomputes_from_its_rows():
    """Where the rows the result records are on disk, its statistics recompute from them
    and P1a's rows exactly, and their scored values have the recorded digests."""
    res = stored_result()
    rows_dir = res["rows"]["dir"]
    rows_dir = rows_dir if os.path.isabs(rows_dir) else os.path.join(E.ROOT, rows_dir)
    if not all(os.path.exists(E.row_path(R, 0, rows=rows_dir)) for R in E.SCORED):
        pytest.skip("the result's rows are not on disk")
    for R in E.REGIMES:
        old, new = E.load_rows(R, rows_dir)
        assert E.scored_digest(new) == res["rows"]["regimes"][R]["scored_values_sha256"]
        summ, _ = E.regime_summary(R, old, new)
        js = (lambda x: json.loads(json.dumps(RS.clean(x))))
        assert js(summ["vs_ship"]) == res["summary"]["regimes"][R]["vs_ship"]
