"""experiments/strong_llm_eval.py on synthetic Kaggle-like shards (no data, no
model): the schema check, the join and its hash verification, the attempt
aggregation, the declared signs, the leave-one-parent-out head, the attempt
call (on every attempted text and on the probe texts only), the gate verdict
and the run's facts; and the entropy job (commit D): its table in the schema
check as its own kind, the join and coverage, its semantics, signs and
registry, that ingesting it leaves every stored result of the rubric and the
attempts as it was (with them in the export, without them, or from a second
directory), its signs stage, reference coverage, verdict, consistency check
and session facts, and the kit's entropy job end to end on its mock backend."""
import hashlib
import json
import math
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import harness as H
from experiments import strong_llm_eval as S
from paiec import llmfeat as F

GROUP = {"matharena": "competition", "multi_swebench": "lang", "real_webagents": "website",
         "researchcodebench": "paper"}


def make_items(n=24, seed=0, benches=S.PARENTS):
    """{bench: {item_id: item dict as load_pairs builds it}} with 4 groups per
    benchmark, and a hidden difficulty per item."""
    rng = np.random.default_rng(seed)
    items, diff = {}, {}
    for b in benches:
        d = {}
        for i in range(n):
            iid = f"{b[:3]}{i:04d}"
            g = f"g{i % 4}" if b != "matharena" else ("aime_2026" if i % 4 == 0 else f"comp{i % 4}")
            feats = f"{GROUP.get(b, 'group')}={g}" + (f";problem_idx={i}" if b == "matharena" else "")
            d[iid] = {"item_content": f"task {i} of {b}: " + "x" * int(rng.integers(20, 200)),
                      "item_features": feats, "interactors": "", "benchmark_id": b}
            diff[(b, iid)] = float(rng.normal())
        items[b] = d
    return items, diff


def write_dir(root, items, rubric_rows, attempt_rows=None, manifest=None, hash_def=S.HASH_DEF):
    os.makedirs(os.path.join(root, "rubric"), exist_ok=True)
    shards = []
    p = os.path.join(root, "rubric", "part-0000.parquet")
    pd.DataFrame(rubric_rows).to_parquet(p, index=False)
    shards.append({"path": "rubric/part-0000.parquet", "rows": len(rubric_rows), "sha256": S.file_sha256(p)})
    if attempt_rows is not None:
        os.makedirs(os.path.join(root, "attempts"), exist_ok=True)
        p = os.path.join(root, "attempts", "part-0000.parquet")
        pd.DataFrame(attempt_rows).to_parquet(p, index=False)
        shards.append({"path": "attempts/part-0000.parquet", "rows": len(attempt_rows),
                       "sha256": S.file_sha256(p)})
    man = {"schema_version": S.SCHEMA_VERSION, "model": {"repo": "org/model", "revision": "abc"},
           "hash": hash_def, "kinds": {"rubric": {"prompt": "p"}, "attempts": {"prompt": "q"}}, "shards": shards}
    man.update(manifest or {})
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    return man


def rubric_rows(items, fn):
    return [{"benchmark": b, "item_id": iid, "content_sha256": S.content_sha256(it), **fn(b, iid)}
            for b, d in items.items() for iid, it in d.items()]


# --- schema ------------------------------------------------------------------------------

def test_content_hash_is_the_text_the_model_reads():
    it = {"item_content": "Solve x", "item_features": "a=1", "interactors": "", "benchmark_id": "matharena"}
    assert S.content_sha256(it) == hashlib.sha256("Solve x\na=1".encode()).hexdigest()
    assert S.content_sha256(it) == hashlib.sha256(F.item_text(it).encode()).hexdigest()
    assert S.content_sha256({**it, "benchmark_id": "other"}) == S.content_sha256(it)
    assert S.content_sha256({**it, "item_features": "a=2"}) != S.content_sha256(it)


def test_declared_signs():
    assert S.declared_sign("rubric_QLq") == (1, "prefix rubric_")
    assert S.declared_sign("rubric_QLq_entropy")[0] == 0
    assert S.declared_sign("time_log_minutes")[0] == 1
    assert S.declared_sign("solve_share")[0] == -1
    assert S.declared_sign("something_else")[0] == 0
    assert S.declared_sign("rubric_hints", {"rubric_hints": -1}) == (-1, "manifest")


def test_schema_accepts_a_well_formed_directory(tmp_path):
    items, _ = make_items(8)
    rows = rubric_rows(items, lambda b, i: {"rubric_a": 1.0, "rubric_a_entropy": 0.3})
    att = [{"benchmark": "matharena", "item_id": iid, "content_sha256": S.content_sha256(it), "attempt": k,
            "answer": "3", "n_tokens": 100, "capped": False, "tok_entropy": 0.5, "tok_lp": -0.4}
           for iid, it in items["matharena"].items() for k in range(2)]
    write_dir(str(tmp_path), items, rows, att)
    rep = S.check_schema(str(tmp_path))
    assert rep["ok"], rep["errors"]
    assert rep["shards"]["rubric/part-0000.parquet"]["features"] == ["rubric_a", "rubric_a_entropy"]
    assert rep["shards"]["attempts/part-0000.parquet"]["kind"] == "attempts"


def test_schema_flags_errors(tmp_path):
    items, _ = make_items(6)
    rows = rubric_rows(items, lambda b, i: {"rubric_a": 1.0})
    rows.append(dict(rows[0]))                              # a duplicate key for rubric_a
    rows.append({**rows[1], "content_sha256": "0" * 64, "rubric_a": None})   # a second hash for one item
    for r in rows[2:4]:
        del r["content_sha256"]                             # rows without a hash: a warning
    att = [{"benchmark": "matharena", "item_id": "mat0000", "content_sha256": "ab" * 32, "attempt": 0,
            "answer": "1", "n_tokens": 5, "capped": False, "tok_entropy": 0.1}]          # no tok_lp
    man = write_dir(str(tmp_path), items, rows, att, hash_def="md5 of something")
    man["shards"].append({"path": "rubric/part-0001.parquet", "rows": 3})
    man["shards"][0]["rows"] += 1
    with open(os.path.join(tmp_path, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    rep = S.check_schema(str(tmp_path))
    errs = "\n".join(rep["errors"])
    assert not rep["ok"]
    assert "hash definition differs" in errs
    assert "rubric/part-0001.parquet: listed in the manifest, not on disk" in errs
    assert "rows, manifest says" in errs
    assert "feature rubric_a: 1 duplicate" in errs
    assert "more than one content_sha256" in errs
    assert "attempt shard lacks tok_lp" in errs
    assert any("without content_sha256" in w for w in rep["warnings"])
    # a directory without a manifest or shards
    empty = tmp_path / "empty"
    empty.mkdir()
    rep = S.check_schema(str(empty))
    assert "manifest.json missing" in rep["errors"] and "no parquet shards" in rep["errors"]


def test_aliases_and_item_ids_lists_are_normalised():
    df = pd.DataFrame({"bench": ["matharena"], "item_ids": [["a", "b"]], "content_hash": ["ff" * 8],
                       "rubric_a": [2.0]})
    out = S._norm(df)
    assert list(out["item_id"]) == ["a", "b"]
    assert list(out["benchmark"]) == ["matharena", "matharena"]
    assert list(out["content_sha256"]) == ["ff" * 8] * 2


def test_unfinished_shards_are_skipped(tmp_path):
    items, _ = make_items(4)
    write_dir(str(tmp_path), items, rubric_rows(items, lambda b, i: {"rubric_a": 1.0}))
    (tmp_path / "rubric" / ".part-0001.parquet").write_bytes(b"partial")
    (tmp_path / "_tmp").mkdir()
    (tmp_path / "_tmp" / "x.parquet").write_bytes(b"partial")
    assert S.shard_files(str(tmp_path)) == ["rubric/part-0000.parquet"]


# --- join ----------------------------------------------------------------------------------

def test_join_verifies_hashes_and_carries_duplicate_texts():
    base = {"item_content": "same text", "item_features": "competition=c1", "interactors": "",
            "benchmark_id": "matharena"}
    items = {"matharena": {"a": dict(base), "b": dict(base),       # b repeats a's text
                           "c": {**base, "item_content": "other"}, "d": {**base, "item_content": "fourth"},
                           "e": {**base, "item_content": "fifth"}},
             "real_webagents": {"w": {**base, "item_content": "web", "benchmark_id": "real_webagents"}}}
    h = {b: {i: S.content_sha256(it) for i, it in d.items()} for b, d in items.items()}
    table = pd.DataFrame([
        {"benchmark": "matharena", "item_id": "a", "content_sha256": h["matharena"]["a"], "rubric_x": 2.0},
        {"benchmark": "matharena", "item_id": "c", "content_sha256": "0" * 64, "rubric_x": 3.0},       # wrong
        {"benchmark": "matharena", "item_id": "d", "content_sha256": None, "rubric_x": 4.0},           # missing
        {"benchmark": "matharena", "item_id": "e", "content_sha256": h["matharena"]["e"][:16], "rubric_x": 5.0},
        {"benchmark": "matharena", "item_id": "zz", "content_sha256": "ab" * 32, "rubric_x": 1.0},     # unknown
        {"benchmark": "real_webagents", "item_id": "w", "content_sha256": h["real_webagents"]["w"][:8],
         "rubric_x": 6.0},                                                                          # too short
        {"benchmark": "mmdocrag", "item_id": "q", "content_sha256": "ab" * 32, "rubric_x": 1.0},
    ])
    joined, rep = S.join(table, items)
    r = rep["per_benchmark"]["matharena"]
    assert (r["rows"], r["hash_ok"], r["hash_mismatch"], r["hash_missing"], r["unknown_item"]) == (5, 2, 1, 1, 1)
    assert (r["covered_direct"], r["covered_duplicate"], r["covered_items"]) == (2, 1, 3)
    assert rep["per_benchmark"]["real_webagents"]["hash_mismatch"] == 1
    assert rep["unknown_benchmark_rows"] == 1
    got = joined.set_index("item_id")
    assert list(got.index) == ["a", "b", "e"]
    assert got.loc["b", "source"] == "duplicate" and got.loc["b", "rubric_x"] == 2.0
    assert got.loc["a", "source"] == "direct" and got.loc["e", "rubric_x"] == 5.0
    # the keys a run-time predictor would look up
    assert got.loc["a", "key"] == F.key_for(items["matharena"]["a"], "matharena")
    assert got.loc["a", "key"] == got.loc["b", "key"]
    assert got.loc["a", "text_key"] == F.text_key(items["matharena"]["a"])


def test_join_counts_duplicate_texts_that_disagree():
    base = {"item_content": "t", "item_features": "", "interactors": "", "benchmark_id": "matharena"}
    items = {"matharena": {"a": dict(base), "b": dict(base)}}
    hh = S.content_sha256(base)
    table = pd.DataFrame([{"benchmark": "matharena", "item_id": "a", "content_sha256": hh, "att_cot_top_share": 1.0},
                          {"benchmark": "matharena", "item_id": "b", "content_sha256": hh, "att_cot_top_share": 0.5}])
    joined, rep = S.join(table, items)
    assert rep["per_benchmark"]["matharena"]["duplicate_texts_disagreeing"] == 1
    assert list(joined["att_cot_top_share"]) == [1.0, 0.5]         # each keeps its own value


# --- attempts --------------------------------------------------------------------------------

def attempt_rows(b, iid, item, answers, design=None, ent=0.5, capped=False, n_tokens=100):
    out = []
    for k, a in enumerate(answers):
        r = {"benchmark": b, "item_id": iid, "content_sha256": S.content_sha256(item), "attempt": k, "answer": a,
             "n_tokens": n_tokens + k, "capped": capped, "tok_entropy": ent + 0.1 * k, "tok_lp": -ent,
             "lp_answer": -0.2}
        if design:
            r["design"] = design
        out.append(r)
    return out


def test_attempts_aggregate_like_the_attempt_probe():
    it = {"item_content": "p", "item_features": "", "interactors": "", "benchmark_id": "matharena"}
    rows = attempt_rows("matharena", "a", it, ["12", "12", "7", None]) \
        + attempt_rows("matharena", "a", it, ["3", "3"], design="short", capped=True) \
        + attempt_rows("matharena", "b", it, ["1", "2", "3", "4"])
    frames = [("attempts/x.parquet", "attempts", S._norm(pd.DataFrame(rows)))]
    t = S.attempt_table(frames, gold={("matharena", "a"): "12"}).set_index("item_id")
    a = t.loc["a"]
    assert a["att_cot_top_share"] == pytest.approx(0.5)
    p = np.array([2, 1, 1]) / 4
    assert a["att_cot_ans_entropy"] == pytest.approx(float(-(p * np.log(p)).sum()))
    assert a["att_cot_graded"] == pytest.approx(0.5) and a["att_cot_top_correct"] == 1.0
    assert a["att_cot_fail_rate"] == pytest.approx(0.25)
    assert a["att_cot_tok_entropy"] == pytest.approx(0.5 + 0.1 * 1.5)
    assert a["att_cot_mean_len"] == pytest.approx(101.5) and a["att_cot_k"] == 4
    assert a["att_short_trunc_rate"] == 1.0 and a["att_short_top_share"] == 1.0
    assert math.isnan(t.loc["b", "att_cot_graded"])            # no reference answer: no grade
    assert S.attempt_designs(t.columns)["short"]["top_share"] == "att_short_top_share"


def test_attempt_designs_parse_underscored_names():
    cols = ["att_cot_long_lp_answer_top", "att_cot_long_lp_answer", "att_x_k", "att_x_n_distinct", "rubric_a"]
    d = S.attempt_designs(cols)
    assert d == {"cot_long": {"lp_answer_top": "att_cot_long_lp_answer_top", "lp_answer": "att_cot_long_lp_answer"},
                 "x": {"k": "att_x_k", "n_distinct": "att_x_n_distinct"}}


def test_combine_drops_attempts_whose_hash_conflicts():
    items = pd.DataFrame([{"benchmark": "matharena", "item_id": "a", "content_sha256": "aa" * 32, "rubric_x": 1.0},
                          {"benchmark": "matharena", "item_id": "b", "content_sha256": "bb" * 32, "rubric_x": 2.0}])
    att = pd.DataFrame([{"benchmark": "matharena", "item_id": "a", "content_sha256": "cc" * 32,
                         "att_cot_top_share": 1.0},
                        {"benchmark": "matharena", "item_id": "b", "content_sha256": "bb" * 32,
                         "att_cot_top_share": 0.5},
                        {"benchmark": "matharena", "item_id": "c", "content_sha256": "dd" * 32,
                         "att_cot_top_share": 0.25}])
    m, conflicts = S.combine(items, att)
    m = m.set_index("item_id")
    assert conflicts == 1 and math.isnan(m.loc["a", "att_cot_top_share"]) and m.loc["a", "rubric_x"] == 1.0
    assert m.loc["b", "att_cot_top_share"] == 0.5 and m.loc["c", "content_sha256"] == "dd" * 32


def test_pipeline_from_a_directory(tmp_path):
    """load_shards -> check_schema -> item_table + attempt_table -> combine -> join."""
    items, diff = make_items(10)
    rows = rubric_rows(items, lambda b, i: {"rubric_a": diff[(b, i)], "rubric_b": 1.0 - diff[(b, i)]})
    att = [r for iid, it in items["matharena"].items() for r in attempt_rows("matharena", iid, it, ["1", "1"])]
    write_dir(str(tmp_path), items, rows, att)
    loaded = S.load_shards(str(tmp_path))
    assert S.check_schema(str(tmp_path), loaded)["ok"]
    table, conflicts = S.combine(S.item_table(loaded[1]), S.attempt_table(loaded[1]))
    joined, rep = S.join(table, items)
    assert conflicts == 0 and rep["totals"]["hash_ok"] == 40 and len(joined) == 40
    mh = joined[joined["benchmark"] == "matharena"]
    assert np.isfinite(mh["att_cot_top_share"]).all()
    assert not np.isfinite(joined.loc[joined["benchmark"] != "matharena", "att_cot_top_share"]).any()


# --- registry, composites --------------------------------------------------------------------

def test_registry_orients_and_flags_features():
    j = pd.DataFrame({"benchmark": ["matharena"] * 3, "item_id": ["a", "b", "c"], "key": "k", "key_official": "k",
                      "text_key": "t", "content_sha256": "h", "source": "direct",
                      "rubric_QLq": [1.0, 2.0, 3.0], "rubric_QLq_entropy": [0.1, 0.2, 0.3],
                      "rubric_hints": [0.0, 1.0, 2.0], "solve_share": [0.9, 0.5, 0.1], "other": [1.0, 1.0, 2.0],
                      "att_cot_tok_entropy": [0.1, 0.2, 0.3], "att_cot_top_share": [1.0, 0.5, 0.25],
                      "att_cot_graded": [1.0, 0.0, 0.0], "att_cot_k": [4.0, 4.0, 4.0]})
    reg = S.registry(j, {"kinds": {"rubric": {"signs": {"rubric_hints": -1}}}})
    assert set(reg) == {"rubric_QLq", "rubric_QLq_entropy", "rubric_hints", "solve_share", "other",
                        "att_cot_tok_entropy", "att_cot_top_share", "att_cot_graded", "att_cot_k"}
    assert reg["rubric_QLq"]["sign"] == 1 and reg["rubric_QLq"]["usable"]
    assert reg["rubric_QLq_entropy"]["sign"] == 0 and not reg["rubric_QLq_entropy"]["usable"]
    assert reg["rubric_hints"]["sign"] == -1 and reg["rubric_hints"]["source"] == "manifest"
    assert reg["solve_share"]["sign"] == -1 and reg["other"]["sign"] == 0
    assert reg["att_cot_top_share"]["sign"] == -1 and reg["att_cot_tok_entropy"]["sign"] == 1
    assert not reg["att_cot_graded"]["label_free"] and not reg["att_cot_k"]["usable"]
    m = S.oriented_maps(j, reg)
    assert m["solve_share"] == {"a": -0.9, "b": -0.5, "c": -0.1}       # + = harder
    assert m["rubric_hints"]["c"] == -2.0 and m["other"]["c"] == 2.0     # no sign: raw
    assert S.rubric_levels(reg) == ["rubric_QLq", "rubric_hints"]
    names = S.head_specs(reg)
    assert names[S.PRIMARY_HEAD] == (["rubric_QLq", "rubric_hints"], True, False)
    assert names[S.POSFREE_HEAD] == (["rubric_QLq", "rubric_hints"], False, True)     # beside the primary
    assert names["rubric_all_ridge"][0] == ["rubric_QLq", "rubric_hints", "rubric_QLq_entropy"]
    assert names["judge_ridge"][0] == ["rubric_QLq", "rubric_hints", "solve_share"]
    # neither the raw entropy nor the fixed window: no attempt primary
    assert not any(v.get("primary") for v in reg.values())


def test_rubric_sum_is_standardised_within_benchmark():
    rng = np.random.default_rng(3)
    n = 30
    j = pd.DataFrame({"benchmark": ["matharena"] * n + ["multi_swebench"] * n,
                      "item_id": [f"i{k}" for k in range(2 * n)],
                      "rubric_a": np.r_[rng.normal(0, 1, n), rng.normal(5, 3, n)],
                      "rubric_b": np.r_[rng.normal(2, 1, n), rng.normal(-4, 2, n)]})
    reg = S.registry(j)
    rs = S.rubric_sum(j, reg)
    x = np.array([rs[f"i{k}"] for k in range(2 * n)])
    assert abs(x[:n].mean()) < 1e-9 and abs(x[n:].mean()) < 1e-9
    assert S.rubric_sum(j[["benchmark", "item_id", "rubric_a"]], S.registry(j[["benchmark", "item_id",
                                                                                 "rubric_a"]])) == {}


# --- signs and heads -----------------------------------------------------------------------

def planted(items, diff, noise, seed, benches=S.PARENTS, slope=1.0):
    rng = np.random.default_rng(seed)
    return {i: slope * diff[(b, i)] + noise * rng.normal() for b in benches for i in items[b]}


def test_sign_entry_reads_the_declared_sign_on_every_parent():
    items, diff = make_items(40, seed=1)
    target = {b: {i: diff[(b, i)] for i in items[b]} for b in S.PARENTS}
    good = S.sign_entry(planted(items, diff, 0.5, 2), target, items, boots=60, seed=0, declared=1)
    a = good["agreement_within"]
    assert (a["units"], a["positive"], a["declared_sign_ok"]) == (4, 4, True)
    assert good["sign_rule"]["transferred_allowed"] and good["sign_rule_within"]["declared_sign_units"] == 4
    assert good["units"]["matharena"]["spearman_within"]["est"] > 0.5
    assert good["units"]["matharena"]["spearman_within"]["ci_group"] is not None
    bad = S.sign_entry(planted(items, diff, 0.5, 2, slope=-1.0), target, items, boots=60, seed=0, declared=1)
    assert bad["agreement_within"]["positive"] == 0 and not bad["agreement_within"]["declared_sign_ok"]
    none = S.sign_entry(planted(items, diff, 0.5, 2), target, items, boots=60, seed=0, declared=0)
    assert not none["sign_rule"]["transferred_allowed"]


def test_head_is_leave_one_parent_out_and_predicts_other_benchmarks():
    benches = S.PARENTS + ("swe_rebench",)
    items, diff = make_items(40, seed=4, benches=benches)
    rng = np.random.default_rng(5)
    rows = []
    for b in benches:
        for i in items[b]:
            d = diff[(b, i)]
            rows.append({"benchmark": b, "item_id": i, "rubric_a": d + 0.6 * rng.normal(),
                         "rubric_b": 0.5 * d + rng.normal(), "rubric_c": rng.normal()})
    j = pd.DataFrame(rows)
    cols = ["rubric_a", "rubric_b", "rubric_c"]
    target = {b: {i: diff[(b, i)] for i in items[b]} for b in S.PARENTS}
    entry, preds = S.fit_head(j, cols, target, items, boots=60)
    assert entry["positive_parents"] == 4 and entry["mean_pearson"] > 0.4 and entry["r_prong"]
    assert set(entry["folds"]) == set(S.PARENTS)
    assert all(i in preds for i in items["swe_rebench"])            # the all-parent fit
    assert entry["predicted_other_benchmarks"] == 40
    # a parent's out-of-fold predictions never read its own targets
    shuffled = dict(target)
    vals = list(target["matharena"].values())
    shuffled["matharena"] = dict(zip(target["matharena"], np.random.default_rng(9).permutation(vals)))
    _, preds2 = S.fit_head(j, cols, shuffled, items, boots=60)
    for i in items["matharena"]:
        assert preds2[i] == pytest.approx(preds[i], abs=1e-12)
    assert any(abs(preds2[i] - preds[i]) > 1e-9 for i in items["multi_swebench"])
    # fewer than three parents with targets: no head
    entry, preds = S.fit_head(j, cols, {b: target[b] for b in S.PARENTS[:2]}, items, boots=60)
    assert "skipped" in entry and preds == {}


def test_attempt_report_and_call_on_planted_attempts():
    items, diff = make_items(48, seed=6, benches=("matharena",))
    rng = np.random.default_rng(7)
    rows = [{"benchmark": "matharena", "item_id": i, "att_cot_tok_entropy": diff[("matharena", i)]
             + 0.3 * rng.normal(), "att_cot_top_share": rng.uniform(), "att_cot_graded": 0.4, "att_cot_k": 4.0}
            for i in items["matharena"]]
    j = pd.DataFrame(rows)
    target = {"matharena": {i: diff[("matharena", i)] for i in items["matharena"]}}
    rep = S.attempt_report(j, target, items, boots=100)
    v = rep["cot"]
    assert v["n_items"] == 48 and v["competitions"] == 4 and v["accuracy"] == pytest.approx(0.4)
    assert v["features_vs_honest"]["tok_entropy"]["rho"] > 0.6
    assert v["by_year_honest"]["tok_entropy"]["2026"]["n"] == 12
    call = S.attempt_call(rep)
    assert call["call"] == "GO" and call["best_feature"] == "tok_entropy"


def _row(rho, lo, y26=None):
    return {"rho": rho, "ci": [lo, rho + 0.1], "n": 100}


@pytest.mark.parametrize("acc,rho,lo,y26,others,call", [
    (0.05, 0.6, 0.4, 0.5, 0.1, "FLOOR"),                 # FLOOR is checked first
    (0.50, 0.40, 0.20, 0.30, 0.10, "GO"),
    (0.50, 0.40, 0.20, 0.20, 0.10, "WEAK (between KILL and GO)"),   # the 2026 contests fall short
    (0.50, 0.40, 0.10, 0.30, 0.10, "WEAK (between KILL and GO)"),   # the interval reaches 0.10
    (0.50, 0.14, 0.00, 0.30, 0.10, "KILL"),
])
def test_attempt_call_rules(acc, rho, lo, y26, others, call):
    rep = {"cot": {"accuracy": acc, "features_vs_honest": {"tok_entropy": _row(rho, lo), "top_share": _row(others, 0)},
                   "by_year_honest": {"tok_entropy": {"2026": _row(y26, 0)}, "top_share": {"2026": None}}}}
    assert S.attempt_call(rep)["call"] == call


# --- the gate and the verdict ------------------------------------------------------------------

def compact(tl=-0.003, mix=-0.004, worst=0.001, r1b=0.0, r1p=-0.001, on=4):
    return {"tl": tl, "mix": mix, "tl_worst_parent": worst, "r1b": r1b, "r1p": r1p, "folds_on": on}


def full(c):
    return {"folds_on": c["folds_on"],
            "regimes": {"tl": {"est": c["tl"], "worst_parent": c["tl_worst_parent"]}, "mix": {"est": c["mix"]},
                        "r1b": {"est": c["r1b"]}, "r1p": {"est": c["r1p"]}}}


@pytest.mark.parametrize("kw,ok", [({}, True), ({"tl": -0.0015}, False), ({"worst": 0.003}, False),
                                   ({"r1b": 0.002}, False), ({"on": 2}, False), ({"mix": 0.001}, False),
                                   ({"on": None}, None)])
def test_gate_checks_agree_with_the_harness_gate(kw, ok):
    c = compact(**kw)
    mine, theirs = S.gate_checks(c), H.gate(full(c))
    assert mine["pass"] == theirs["pass"] == ok
    assert mine["pass_if_on"] == theirs["pass_if_on"]


def test_verdict_keeps_only_a_declared_primary_passing_both_prongs():
    def entry(**kw):
        lines = {n: {**compact(tl=0.0, on=0), "gate_pass": False} for n in S.NESTED}
        lines["transferred nested"] = {**compact(**kw), "gate_pass": None}
        lines["per-pair s=0.5 from B7 (forced)"] = {**compact(on=None), "gate_pass": None}
        return {"lines": lines}
    all_four = {"positive": 4, "units": 4, "declared_sign_ok": True}
    three = {"positive": 3, "units": 4, "declared_sign_ok": True}
    state = {
        "harness": {"_meta": {}, "head rubric_ridge": entry(), "head judge_ridge": entry(),
                    "head rubric_ridge_posfree": entry(), "rubric_QLq": entry(),
                    "att_cot_ent_first1024": entry(tl=-0.001), "rubric_VO": entry(worst=0.004),
                    "rubric_pos": entry()},
        "heads": {"rubric_ridge": {"positive_parents": 4, "positive_parents_prong": 3, "mean_pearson": 0.25},
                  "judge_ridge": {"positive_parents": 3, "positive_parents_prong": 2, "mean_pearson": 0.2},
                  "rubric_ridge_posfree": {"positive_parents": 3, "positive_parents_prong": 3}},
        "signs": {"registry": {"rubric_QLq": {"kind": "rubric"}, "rubric_VO": {"kind": "rubric"},
                               "rubric_pos": {"kind": "rubric"},
                               "att_cot_ent_first1024": {"kind": "attempt", "feature": "ent_first1024",
                                                         "primary": True}},
                  "features": {"rubric_QLq": {"agreement_prong": three, "agreement_within": all_four},
                               "rubric_VO": {"agreement_prong": all_four, "agreement_within": all_four},
                               # positive within competition only through position: not the prong
                               "rubric_pos": {"agreement_prong": {"positive": 2, "units": 4,
                                                                  "declared_sign_ok": False},
                                              "agreement_within": all_four},
                               "att_cot_ent_first1024": {"agreement_prong": {"positive": 1, "units": 1,
                                                                             "declared_sign_ok": False}}}},
        "ingest": {"attempt_semantics": {"primary_attempt": "ent_first1024", "logprobs": "processed"}}}
    v = S.verdict(state)
    assert v["keep"] == ["head rubric_ridge"] and v["call"] == "KEEP head rubric_ridge"
    assert v["exploratory_pass"] == ["head rubric_ridge_posfree", "rubric_QLq"]
    assert v["posfree_head"] == {"gate_pass_nested": True, "sign_prong": True, "best_nested_tl": -0.003}
    assert v["attempt_primary"] == "ent_first1024" and v["attempt_logprobs"] == "processed"
    f = v["features"]
    # judge_ridge: positive r on three parents, but not on matharena net of position
    assert f["head judge_ridge"]["gate_pass_nested"] and not f["head judge_ridge"]["sign_prong"]
    assert f["att_cot_ent_first1024"]["primary"] and not f["att_cot_ent_first1024"]["keep"]
    assert not f["rubric_VO"]["gate_pass_nested"]
    assert f["rubric_pos"]["gate_pass_nested"] and not f["rubric_pos"]["sign_prong"]
    assert f["head rubric_ridge"]["forced_per_pair_tl"] == {"per-pair s=0.5 from B7 (forced)": -0.003}
    state["harness"]["head rubric_ridge"] = entry(tl=-0.0019)
    assert S.verdict(state)["call"].startswith("NULL")
    state["harness"]["head rubric_ridge"] = entry()
    state["heads"]["rubric_ridge"]["positive_parents_prong"] = 2       # matharena fails net of position
    assert S.verdict(state)["call"].startswith("NULL")


# --- the notebook's extras (kaggle/strong_probe/strong_probe.py) ----------------------------------

def k1_attempt_rows(b, iid, item, n=3):
    """An attempt shard row set with every column the notebook's export writes."""
    rows = []
    for k in range(n):
        rows.append({"benchmark": b, "item_id": iid, "content_sha256": S.content_sha256(item), "attempt": k,
                     "design": "cot", "answer": None if k == 2 else "5", "n_tokens": 1000 + k, "capped": k == 2,
                     "tok_entropy": 0.4, "tok_lp": -0.3, "lp_answer": float("nan") if k == 2 else -0.1,
                     "refuse": False, "forced": k == 2, "closed": k != 2, "n_think": 900 + k,
                     "tok_entropy_think": 0.5 + k, "tok_lp_think": -0.5, "tok_entropy_answer": 0.1,
                     "tok_lp_answer": -0.05, "ent_first256": 0.6, "ent_first1024": 0.55 + 0.1 * k,
                     "lp_first256": -0.6, "lp_first1024": -0.55})
    return rows


def test_notebook_attempt_columns_pass_the_schema_and_are_averaged(tmp_path):
    items, _ = make_items(4)
    att = [r for iid, it in items["matharena"].items() for r in k1_attempt_rows("matharena", iid, it)]
    write_dir(str(tmp_path), items, rubric_rows(items, lambda b, i: {"rubric_reasoning": 2.0}), att)
    rep = S.check_schema(str(tmp_path))
    assert rep["ok"] and not rep["warnings"], (rep["errors"], rep["warnings"])
    t = S.attempt_table(S.load_shards(str(tmp_path))[1]).set_index("item_id")
    a = t.loc["mat0000"]
    assert a["att_cot_closed_rate"] == pytest.approx(2 / 3) and a["att_cot_forced_rate"] == pytest.approx(1 / 3)
    assert a["att_cot_mean_think_len"] == pytest.approx(901) and a["att_cot_tok_entropy_think"] == pytest.approx(1.5)
    assert a["att_cot_ent_first1024"] == pytest.approx(0.65) and a["att_cot_lp_answer"] == pytest.approx(-0.1)
    j = t.reset_index()
    reg = S.registry(j)
    assert reg["att_cot_closed_rate"]["sign"] == -1 and reg["att_cot_ent_first256"]["sign"] == 1
    assert reg["att_cot_lp_first1024"]["sign"] == -1 and reg["att_cot_forced_rate"]["sign"] == 1
    assert reg["att_cot_tok_lp_answer"]["sign"] == -1 and reg["att_cot_lp_answer"]["sign"] == -1
    assert reg["att_cot_ent_first256"]["source"] == "ATTEMPT_EXTRA"
    # a column nobody reads is reported
    extra = pd.DataFrame(att).assign(mystery=1.0)
    p = tmp_path / "attempts" / "part-0000.parquet"
    extra.to_parquet(p, index=False)
    man = json.load(open(tmp_path / "manifest.json"))
    man["shards"][1]["sha256"] = S.file_sha256(str(p))
    json.dump(man, open(tmp_path / "manifest.json", "w"))
    rep = S.check_schema(str(tmp_path))
    assert rep["ok"] and any("attempt columns not read: ['mystery']" in w for w in rep["warnings"])


def test_notebook_rubric_names_carry_the_notebooks_signs():
    scales = ("reasoning", "knowledge", "work", "interaction", "volume", "atypicality", "precision",
              "unguessability")
    for s in scales:
        assert S.declared_sign(f"rubric_{s}")[0] == 1
        assert S.declared_sign(f"rubric_{s}_entropy")[0] == 0 and S.declared_sign(f"rubric_{s}_mass")[0] == 0
    for name, sign in (("solve_share", -1), ("time_log_minutes", 1), ("solve_share_mass", 0),
                       ("time_log_minutes_entropy", 0), ("prompt_tokens", 0), ("gen_tokens", 0),
                       ("truncated_n", 0)):
        assert S.declared_sign(name)[0] == sign


# --- what the attempt log-probs are (vLLM V0 returns them after the penalties and sampling) ------

V0 = "processed (V0): after temperature, penalties and top-k/top-p; greedy requests get the raw distribution"


def _frames(rows):
    return [("attempts/x.parquet", "attempts", S._norm(pd.DataFrame(rows)))]


def test_logprob_semantics_follow_the_manifest():
    assert S.logprob_semantics({"logprobs": V0}) == "processed"
    assert S.logprob_semantics({"logprobs": "processed_topk"}) == "processed"
    assert S.logprob_semantics({"logprobs": "raw"}) == "raw"                   # the notebook's recorder
    assert S.logprob_semantics({"logprobs": "raw_topk"}) == "raw_topk"
    assert S.logprob_semantics({"logprobs": "raw model distribution (V1)"}) == "raw_topk"   # an older export
    assert S.logprob_semantics({"logprobs": "mixed"}) == "mixed"
    assert S.logprob_semantics({}) == S.logprob_semantics(None) == "unknown"
    failed = {"logprobs": "raw", "kinds": {"attempts": {"token_stats": {"recorder_check": {"status": "FAILED"}}}}}
    assert S.logprob_semantics(failed) == "unknown"


def test_an_answer_log_prob_mixing_readouts_is_found():
    it = {"item_content": "p", "item_features": "", "interactors": "", "benchmark_id": "matharena"}
    rows = k1_attempt_rows("matharena", "a", it)
    rows[2]["lp_answer"] = -0.9                     # the forced (greedy, raw) readout of attempt 2
    man = {"logprobs": V0}
    plan = S.lp_answer_plan(_frames(rows), man)
    assert not plan["uniform"] and plan["column"] == "lp_answer" and (plan["sampled"], plan["forced"]) == (2, 1)
    # raw log-probs, the manifest's promise, or one kind of readout: uniform
    assert S.lp_answer_plan(_frames(rows), {"logprobs": "raw model distribution (V1)"})["uniform"]
    assert S.lp_answer_plan(_frames(rows), {**man, "kinds": {"attempts": {"force_every_attempt": True}}})["uniform"]
    k1 = {"attempts": {"forced": "every attempt: a greedy continuation of at most 40 tokens after its reasoning"}}
    assert S.lp_answer_plan(_frames(rows), {**man, "kinds": k1})["uniform"]
    assert not S.lp_answer_plan(_frames(rows), {**man, "kinds": {"attempts": {"forced": "none (--no-force)"}}})[
        "uniform"]
    assert not S.lp_answer_plan(_frames(rows), {"logprobs": "mixed"})["uniform"]
    assert S.lp_answer_plan(_frames(k1_attempt_rows("matharena", "a", it)), man)["uniform"]   # forced ones NaN
    # no forced flags: a processed readout cannot be told apart from a forced one
    bare = [{k: v for k, v in r.items() if k != "forced"} for r in rows]
    assert not S.lp_answer_plan(_frames(bare), man)["uniform"]
    # lp_forced (the forced readout for every attempt) is the answer log-prob
    forced_all = [{**r, "lp_forced": -0.5 - 0.1 * r["attempt"]} for r in rows]
    plan = S.lp_answer_plan(_frames(forced_all), man)
    assert plan["uniform"] and plan["column"] == "lp_forced"
    t = S.attempt_table(_frames(forced_all), lp_col=plan["column"]).set_index("item_id")
    assert t.loc["a", "att_cot_lp_answer"] == pytest.approx(-0.6)


def test_raw_columns_pass_the_schema_and_become_the_primary(tmp_path):
    items, _ = make_items(10)
    att = []
    for iid, it in items["matharena"].items():
        for r in k1_attempt_rows("matharena", iid, it):
            att.append({**r, "tok_entropy_raw": 2.0 + r["attempt"], "tok_lp_raw": -1.0, "ent_first1024_raw": 1.5,
                        "lp_forced": -0.4})
    write_dir(str(tmp_path), items, rubric_rows(items, lambda b, i: {"rubric_reasoning": 2.0}), att,
              manifest={"logprobs": V0})
    rep = S.check_schema(str(tmp_path))
    assert rep["ok"] and not rep["warnings"], (rep["errors"], rep["warnings"])
    man, frames = S.load_shards(str(tmp_path))
    plan = S.lp_answer_plan(frames, man)
    t = S.attempt_table(frames, lp_col=plan["column"])
    assert t["att_cot_tok_entropy_raw"].tolist() == pytest.approx([3.0] * 10)
    assert t["att_cot_lp_answer"].tolist() == pytest.approx([-0.4] * 10)
    sem = S.attempt_semantics(man, t, plan)
    assert sem["primary_attempt"] == "tok_entropy_raw" and sem["d2"]["comparable"]
    assert sem["raw_features"] == ["ent_first1024_raw", "tok_entropy_raw", "tok_lp_raw"]
    reg = S.registry(t, man, sem)
    assert reg["att_cot_tok_entropy_raw"]["primary"] and reg["att_cot_tok_entropy_raw"]["sign"] == 1
    assert reg["att_cot_tok_lp_raw"]["sign"] == -1 and reg["att_cot_tok_lp_raw"]["distribution"] == "raw"
    assert reg["att_cot_tok_entropy"]["distribution"] == "processed" and not reg["att_cot_tok_entropy"]["primary"]
    assert reg["att_cot_lp_answer"]["usable"] and reg["att_cot_lp_answer"]["distribution"].startswith("raw")


def test_the_primary_falls_back_to_a_fixed_window():
    n = 12
    base = {"benchmark": ["matharena"] * n, "item_id": [f"i{k}" for k in range(n)],
            "att_cot_tok_entropy": np.linspace(0, 1, n), "att_cot_ent_first1024": np.linspace(1, 2, n)}
    j = pd.DataFrame(base)
    assert S.primary_attempt(j) == S.primary_attempt(j, "processed") == "ent_first1024"
    # tok_entropy is the primary only when the manifest says it is the raw full-vocabulary entropy
    assert S.primary_attempt(j, "raw") == "tok_entropy"
    for sem in ("raw_topk", "mixed", "unknown"):
        assert S.primary_attempt(j, sem) == "ent_first1024"
    # a raw column the notebook could not fill does not count
    assert S.primary_attempt(j.assign(att_cot_tok_entropy_raw=np.nan)) == "ent_first1024"
    assert S.primary_attempt(j.assign(att_cot_tok_entropy_raw=1.0)) == "tok_entropy_raw"
    assert S.primary_attempt(j.drop(columns="att_cot_ent_first1024")) is None      # never tok_entropy
    ok, why = S.d2_comparable("ent_first1024", "processed")
    assert not ok and "top-5 lower bound" in why
    assert S.d2_comparable("tok_entropy", "raw")[0] and not S.d2_comparable("tok_entropy", "raw_topk")[0]
    assert not S.d2_comparable("ent_first1024", "raw")[0] and not S.d2_comparable(None, "raw")[0]


def test_the_4b_decision_is_withheld_beside_a_processed_primary(monkeypatch):
    monkeypatch.setattr(S, "_ref_4b", lambda: {"call": "FLOOR"})
    proc = {"primary_attempt": "ent_first1024", "logprobs": "processed",
            "d2": dict(zip(("comparable", "reason"), S.d2_comparable("ent_first1024", "processed")))}
    r = S.reference_4b(proc)
    assert not r["comparable"] and r["decision"] is None and "withheld" in r
    raw5 = {**proc, "logprobs": "raw", "d2": dict(zip(("comparable", "reason"), S.d2_comparable("ent_first1024",
                                                                                                 "raw")))}
    r = S.reference_4b(raw5)
    assert not r["comparable"] and r["decision"] == {"call": "FLOOR"}                 # flagged, shown
    full = {**proc, "primary_attempt": "tok_entropy_raw",
            "d2": dict(zip(("comparable", "reason"), S.d2_comparable("tok_entropy_raw", "processed")))}
    r = S.reference_4b(full)
    assert r["comparable"] and r["decision"] == {"call": "FLOOR"}
    for sem in ("mixed", "unknown"):
        assert S.reference_4b({**proc, "logprobs": sem})["decision"] is None
    assert not S.reference_4b(None)["comparable"]


def k1_manifest(logprobs="raw", forced=True):
    """The attempt fields kaggle/strong_probe/strong_probe.py's export manifest carries."""
    return {"logprobs": logprobs, "logprobs_engine": V0,
            "kinds": {"attempts": {
                "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
                "forced": ("every attempt: a greedy continuation of at most 40 tokens after its reasoning"
                           if forced else "none (--no-force)"),
                "token_stats": {"logprobs": logprobs, "comparable_to_d2": logprobs == "raw",
                                "sources": {"raw_full_vocab": 12}, "recorder_check": {"status": "ok"}}}}}


def test_the_notebooks_raw_recorder_export_reads_as_d2():
    """The notebook writes the raw full-vocabulary statistics into the standard
    columns (manifest logprobs 'raw') and forces a readout for every attempt."""
    items, _ = make_items(10)
    att = [r for iid, it in items["matharena"].items() for r in k1_attempt_rows("matharena", iid, it)]
    for r in att:
        r["lp_answer"] = -0.05                           # the forced readout, for every attempt
    frames = _frames(att)
    man = k1_manifest("raw")
    plan = S.lp_answer_plan(frames, man)
    assert plan["uniform"] and "every attempt" in plan["why"]
    t = S.attempt_table(frames, lp_col=plan["column"])
    sem = S.attempt_semantics(man, t, plan)
    assert sem["logprobs"] == "raw" and sem["primary_attempt"] == "tok_entropy" and sem["d2"]["comparable"]
    assert sem["recorder_check"] == "ok" and not sem["notes"]           # no penalty note on raw statistics
    reg = S.registry(t, man, sem)
    assert reg["att_cot_tok_entropy"]["primary"] and reg["att_cot_tok_entropy"]["distribution"] == "raw"
    assert reg["att_cot_lp_answer"]["distribution"] == "raw (greedy forced readout)"
    assert not any(v.get("excluded") for v in reg.values())
    # the same export from mixed sources: the token statistics are excluded, a uniform lp_answer is not
    man = k1_manifest("mixed")
    sem = S.attempt_semantics(man, t, S.lp_answer_plan(frames, man))
    reg = S.registry(t, man, sem)
    assert sem["primary_attempt"] == "ent_first1024" and not sem["d2"]["comparable"]
    assert reg["att_cot_ent_first1024"]["excluded"] and reg["att_cot_tok_entropy"]["excluded"]
    assert reg["att_cot_lp_answer"]["usable"] and reg["att_cot_top_share"]["usable"]
    assert any("presence penalty" in n for n in sem["notes"])


def test_a_mixed_answer_log_prob_is_excluded_from_the_call_and_the_harness():
    items, diff = make_items(48, seed=6, benches=("matharena",))
    rng = np.random.default_rng(7)
    rows = [{"benchmark": "matharena", "item_id": i, "att_cot_lp_answer": -diff[("matharena", i)]
             + 0.1 * rng.normal(), "att_cot_top_share": rng.uniform(), "att_cot_ent_first1024": rng.normal(),
             "att_cot_graded": 0.4, "att_cot_k": 4.0} for i in items["matharena"]]
    j = pd.DataFrame(rows)
    sem = {"logprobs": "processed", "lp_answer": {"uniform": False, "column": "lp_answer", "why": "mixed"}}
    reg = S.registry(j, None, sem)
    assert reg["att_cot_lp_answer"]["excluded"] and not reg["att_cot_lp_answer"]["usable"]
    assert reg["att_cot_lp_answer"]["sign"] == -1 and reg["att_cot_ent_first1024"]["primary"]
    target = {"matharena": {i: diff[("matharena", i)] for i in items["matharena"]}}
    exclude = {v["feature"]: v["excluded"] for v in reg.values() if v.get("excluded")}
    rep = S.attempt_report(j, target, items, boots=100, exclude=exclude, primary="ent_first1024")
    v = rep["cot"]
    assert "lp_answer" not in v["features_vs_honest"] and v["excluded"]["lp_answer"]["vs_honest"]["rho"] > 0.6
    assert "ent_first1024_vs_prompt_chars" in v
    call = S.attempt_call(rep, "ent_first1024")
    assert call["best_feature"] != "lp_answer" and call["call"] != "GO"
    assert call["primary"]["feature"] == "ent_first1024" and "cot" in call["primary"]["designs"]
    # the harness never sees it
    assert "att_cot_lp_answer" not in S.harness_maps("/nonexistent", j, reg)


def test_stage_semantics_require_a_current_ingest():
    j = pd.DataFrame({"benchmark": ["matharena"], "item_id": ["a"], "att_cot_tok_entropy": [0.1]})
    with pytest.raises(SystemExit):
        S.stage_semantics({"ingest": {"features_digest": "x"}}, j)
    assert S.stage_semantics({}, j.drop(columns="att_cot_tok_entropy")) is None


# --- out-of-fold heads are tied to the features they were fitted on -----------------------------------

def test_stale_heads_never_enter_the_harness(tmp_path):
    work = str(tmp_path)
    j = pd.DataFrame({"benchmark": ["matharena"] * 3, "item_id": ["a", "b", "c"], "rubric_a": [1.0, 2.0, 3.0],
                      "rubric_b": [0.0, 1.0, 0.5]})
    j.to_parquet(os.path.join(work, "features.parquet"), index=False)
    dig = S.features_digest(work)
    reg = S.registry(j)
    oof = os.path.join(work, "oof.json")
    H.save_json(oof, {"features_digest": dig, "heads": {"rubric_ridge": {"a": 0.1, "b": 0.2}}})
    assert S.harness_maps(work, j, reg)["head rubric_ridge"] == {"a": 0.1, "b": 0.2}
    H.save_json(oof, {"features_digest": "0" * 16, "heads": {"rubric_ridge": {"a": 0.1}}})
    assert "head rubric_ridge" not in S.harness_maps(work, j, reg) and S.load_oof(work) == {}
    H.save_json(oof, {"rubric_ridge": {"a": 0.1}})                    # the format without a digest
    assert "head rubric_ridge" not in S.harness_maps(work, j, reg)
    # a re-ingest with other features drops the downstream results and oof.json
    H.save_json(oof, {"features_digest": dig, "heads": {"rubric_ridge": {"a": 0.1}}})
    state = {"ingest": {"features_digest": dig}, "meta": {}, "signs": {}, "heads": {}, "harness": {}}
    assert S.drop_stale(dict(state), work, dig) == [] and os.path.exists(oof)       # same features: kept
    dropped = S.drop_stale(state, work, "f" * 16)
    assert set(dropped) == {"signs", "heads", "harness", "oof.json"} and not os.path.exists(oof)
    assert set(state) == {"ingest", "meta"}
    # the same features under other declared log-probs: the downstream results go, oof.json stays
    H.save_json(oof, {"features_digest": dig, "heads": {"rubric_ridge": {"a": 0.1}}})
    sem = {"logprobs": "raw", "notes": []}
    state = {"ingest": {"features_digest": dig, "attempt_semantics": sem}, "signs": {}, "attempts": {}}
    assert S.drop_stale(dict(state), work, dig, semantics=dict(sem)) == []
    assert set(S.drop_stale(state, work, dig, semantics={**sem, "logprobs": "processed"})) == {"signs", "attempts"}
    assert os.path.exists(oof)


# --- position: problem_idx is in the rubric prompt's metadata -------------------------------------------

def test_the_sign_prong_is_net_of_position_on_matharena():
    per = {"matharena": {"applies": True, "spearman_within": {"est": 0.4}, "partial2_spearman": {"est": -0.05}},
           "multi_swebench": {"applies": True, "spearman_within": {"est": 0.2}},
           "real_webagents": {"applies": True, "spearman_within": {"est": 0.1}},
           "researchcodebench": {"applies": True, "spearman_within": {"est": -0.1}}}
    assert S.agreement(per)["positive"] == 3 and S.agreement(per)["declared_sign_ok"]
    a = S.agreement(per, "spearman_within", S.PRONG_FIELD)
    assert a["positive"] == 2 and not a["declared_sign_ok"] and a["fields"]["matharena"] == "partial2_spearman"
    # a missing position-net statistic is not positive (never the statistic that carries position)
    del per["matharena"]["partial2_spearman"]
    a = S.agreement(per, "spearman_within", S.PRONG_FIELD)
    assert a["missing"] == ["matharena"] and a["units"] == 4 and a["positive"] == 2
    # a feature that echoes position and runs against difficulty beyond it: positive within
    # competition, negative net of position
    items, diff = make_items(80, seed=11)
    x = {}
    for i, k in enumerate(items["matharena"]):
        e = np.random.default_rng(i).normal()
        diff[("matharena", k)] = math.log1p(i) + 0.3 * e
        x[k] = math.log1p(i) - 0.15 * e
    target = {b: {i: diff[(b, i)] for i in items[b]} for b in S.PARENTS}
    x.update(planted(items, diff, 0.5, 2, benches=S.PARENTS[1:]))
    e = S.sign_entry(x, target, items, boots=60, seed=0, declared=1)
    m = e["units"]["matharena"]
    assert m["spearman_within"]["est"] > 0.5 and m["partial2_spearman"]["est"] < -0.5
    assert e["agreement_within"]["positive"] == 4 and e["agreement_prong"]["positive"] == 3
    assert e["agreement_prong"]["fields"]["matharena"] == "partial2_spearman"


def test_the_position_free_head_carries_no_position():
    items, diff = make_items(48, seed=12)
    rng = np.random.default_rng(13)
    rows = []
    for b in S.PARENTS:
        for i, k in enumerate(items[b]):
            d = diff[(b, k)]
            pos = math.log1p(i) if b == "matharena" else 0.0
            if b == "matharena":
                diff[(b, k)] = d = pos + 0.5 * d
            rows.append({"benchmark": b, "item_id": k, "rubric_a": 2 * pos + d + 0.5 * rng.normal(),
                         "rubric_b": pos + rng.normal()})
    j = pd.DataFrame(rows)
    cols = ["rubric_a", "rubric_b"]
    bench, iid = j["benchmark"].to_numpy().astype(str), j["item_id"].to_numpy().astype(str)
    Xr, info = S.residualise_position(j[cols].to_numpy(float), bench, iid, items)
    assert set(info) == {"matharena"} and info["matharena"]["rows"] == 48
    m = bench == "matharena"
    comp = np.array([S.IC.features(items["matharena"][k])["competition"] for k in iid[m]])
    pos = np.array([S.IC.position(items["matharena"][k]) for k in iid[m]])

    def dm(v):
        return v - pd.Series(v).groupby(comp).transform("mean").to_numpy()

    for c in range(2):
        assert abs(dm(pos) @ dm(Xr[m, c])) < 1e-8                                   # no position left
        assert dm(pos) @ dm(j[cols].to_numpy(float)[m, c]) > 1.0                    # there was before
        assert pd.Series(Xr[m, c]).groupby(comp).mean().to_numpy() == pytest.approx(
            pd.Series(j[cols].to_numpy(float)[m, c]).groupby(comp).mean().to_numpy())   # group means kept
    assert np.array_equal(Xr[~m], j[cols].to_numpy(float)[~m])                      # no position elsewhere
    target = {b: {k: diff[(b, k)] for k in items[b]} for b in S.PARENTS}
    plain, p0 = S.fit_head(j, cols, target, items, boots=60)
    free, p1 = S.fit_head(j, cols, target, items, boots=60, posfree=True)
    assert free["posfree"] and free["position_residualised"]["matharena"]["rows"] == 48
    o0 = np.array([p0[k] for k in iid[m]])
    o1 = np.array([p1[k] for k in iid[m]])
    assert dm(pos) @ dm(o0) > 0.5 and abs(dm(pos) @ dm(o1)) < 1e-8
    # the head prong reads matharena net of position
    assert plain["prong"]["fields"]["matharena"] == "partial2_pearson"
    assert plain["prong"]["values"]["matharena"] == plain["per_parent"]["matharena"]["partial2_pearson"]["est"]
    assert plain["positive_parents_prong"] == sum(1 for r in plain["prong"]["values"].values() if r and r > 0)


# --- the attempt rule on the probe texts (the units it was fixed for) and the run's facts ------------------

def write_units(kdir, units, extra=None):
    """_detail/attempt_units.parquet as the notebook's export writes it: one row per unique text."""
    d = os.path.join(kdir, "_detail")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([{**u, **(extra(u) if extra else {})} for u in units]).to_parquet(
        os.path.join(d, "attempt_units.parquet"), index=False)


def test_probe_units_read_the_notebooks_flag(tmp_path):
    kdir = str(tmp_path)
    assert S.probe_units(kdir) is None                                  # no table: no flag
    units = [{"unit": "u0", "benchmark": "matharena", "item_ids": ["a", "b"], "probe": True},
             {"unit": "u1", "benchmark": "matharena", "item_ids": ["c"], "probe": True},
             {"unit": "u2", "benchmark": "matharena", "item_ids": ["d", "e", "f"], "probe": False},
             {"unit": "u3", "benchmark": "matharena", "item_ids": ["c"], "probe": False}]   # c under both
    write_units(kdir, units)
    pu = S.probe_units(kdir)
    assert pu["probe"] == {("matharena", "a"), ("matharena", "b")}
    assert pu["rest"] == {("matharena", k) for k in "def"} and pu["conflicting_items"] == 1
    assert pu["units"] == {"probe": 2, "rest": 2} and pu["items"] == {"probe": 2, "rest": 3}
    assert pu["path"] == S.UNITS_DETAIL and pu["probe_ids_check"] is None      # unchecked without PROBE_IDS
    with open(os.path.join(kdir, S.UNITS_DETAIL), "rb") as fh:
        assert pu["sha256"] == hashlib.sha256(fh.read()).hexdigest()           # the manifest does not hash it
    # a text is a probe text iff one of its item_ids is listed: u0 (a listed), u1 (c listed; c's other text u3
    # is unflagged, so the check reads per unit)
    ok = S.probe_units(kdir, frozenset({"a", "c", "zz"}))["probe_ids_check"]
    assert ok["units_flag_mismatch"] == 1 and not ok["ok"]                    # u3 holds listed c, unflagged
    ok = S.probe_units(kdir, frozenset({"b", "c"}))["probe_ids_check"]
    assert ok["units_checked"] == 4 and ok["listed_in_flagged_units"] == 2 and ok["listed_in_unflagged_units"] == 1
    write_units(kdir, units[:3])
    good = S.probe_units(kdir, frozenset({"a", "c", "zz"}))["probe_ids_check"]
    assert good["ok"] and good["units_flag_mismatch"] == 0 and good["listed_not_attempted"] == 1
    bad = S.probe_units(kdir, frozenset({"a", "d"}))["probe_ids_check"]        # u1 flagged, none listed; u2 listed
    assert not bad["ok"] and bad["units_flag_mismatch"] == 2
    write_units(kdir, units)
    write_units(kdir, [{k: v for k, v in u.items() if k != "probe"} for u in units])
    assert S.probe_units(kdir) is None                                  # no flag column
    j = pd.DataFrame({"benchmark": ["matharena", "matharena", "other"], "item_id": ["a", "z", "a"]})
    assert S.unit_rows(j, pu["probe"])["item_id"].tolist() == ["a"]     # keyed on (benchmark, item_id)


def _probe_fixture(tmp_path, monkeypatch, n=48):
    """A joined table with planted attempts (the probe texts' tok_entropy follows the
    difficulty, the others' is noise), its ingest state, and the stage's inputs stubbed."""
    items, diff = make_items(n, seed=21, benches=("matharena",))
    rng = np.random.default_rng(22)
    ids = sorted(items["matharena"])
    probe = set(ids[: 2 * n // 3])
    rows = []
    for i in ids:
        d = diff[("matharena", i)]
        ent = d + 0.2 * rng.normal() if i in probe else rng.normal()
        rows.append({"benchmark": "matharena", "item_id": i, "key": i, "key_official": i, "text_key": i,
                     "content_sha256": "h", "source": "direct", "rubric_a": rng.normal(),
                     "att_cot_tok_entropy": ent, "att_cot_top_share": rng.uniform(), "att_cot_graded": 0.4,
                     "att_cot_k": 4.0})
    work, kdir = tmp_path / "work", tmp_path / "kaggle"
    work.mkdir()
    kdir.mkdir()
    pd.DataFrame(rows).to_parquet(work / "features.parquet", index=False)
    units = [{"unit": f"u{k}", "benchmark": "matharena", "item_ids": [i], "probe": i in probe}
             for k, i in enumerate(ids)]
    write_units(str(kdir), units)
    out = str(tmp_path / "out.json")
    sem = {"primary_attempt": "tok_entropy", "logprobs": "raw", "lp_answer": {"uniform": True}, "notes": [],
           "d2": {"comparable": True, "reason": "test"}}
    H.save_json(out, {"ingest": {"features_digest": S.features_digest(str(work)), "attempt_semantics": sem},
                      "meta": {"manifest": {}}})
    target = {"matharena": {i: diff[("matharena", i)] for i in ids}}
    monkeypatch.setattr(S, "targets", lambda work, refresh=False: (target, None, None))
    monkeypatch.setattr(S.ICE, "load_items", lambda b: items[b])
    monkeypatch.setattr(S, "ROOT", str(tmp_path))          # no strong-tier targets, no 4B decision
    return {"work": str(work), "kaggle": str(kdir), "out": out, "probe": probe, "rows": rows}


def _args(fx, units):
    from types import SimpleNamespace
    return SimpleNamespace(work=fx["work"], kaggle=fx["kaggle"], out=fx["out"], boots=100, attempt_units=units)


def test_the_attempt_rule_on_the_probe_texts_sits_beside_the_all_units_reading(tmp_path, monkeypatch):
    fx = _probe_fixture(tmp_path, monkeypatch)
    S.stage_attempts(_args(fx, "all"))
    st = H.load_json(fx["out"])
    base = st["attempts"]
    assert "probe_only" not in base and base["designs"]["cot"]["n_items"] == 48
    # the probe reading leaves every all-units field as it was, bit for bit
    S.stage_attempts(_args(fx, "probe"))
    at = H.load_json(fx["out"])["attempts"]
    assert {k: v for k, v in at.items() if k != "probe_only"} == base
    po = at["probe_only"]
    assert po["designs"]["cot"]["n_items"] == len(fx["probe"]) == 32
    dig, sdig = S.features_digest(fx["work"]), H.digest(["experiments/strong_llm_eval.py"])
    assert base["features_digest"] == dig and base["script_digest"] == sdig and po["script_digest"] == sdig
    with open(os.path.join(fx["kaggle"], S.UNITS_DETAIL), "rb") as fh:
        assert po["source_sha256"] == hashlib.sha256(fh.read()).hexdigest()
    assert po["probe_ids_check"] is None                    # no strong_probe.py under this ROOT
    assert po["flagged_units"] == {"probe": 32, "rest": 16} and po["flagged_items"] == {"probe": 32, "rest": 16}
    assert po["features_digest"] == S.features_digest(fx["work"])
    rho_p = po["designs"]["cot"]["features_vs_honest"]["tok_entropy"]["rho"]
    rho_a = base["designs"]["cot"]["features_vs_honest"]["tok_entropy"]["rho"]
    assert rho_p > rho_a and po["decision"]["call"] == "GO" and po["decision"]["primary"]["feature"] == "tok_entropy"
    # the same rows through attempt_report directly
    j = S.load_joined(fx["work"])
    rep = S.attempt_report(S.unit_rows(j, {("matharena", i) for i in fx["probe"]}), S.targets(None)[0],
                           {"matharena": S.ICE.load_items("matharena")}, 100, None, exclude={}, primary="tok_entropy")
    assert rep == po["designs"]
    # the length partial and its unadjusted counterpart are one statistic (corr_block), not the rule's
    c = rep["cot"]
    assert set(c["spearman_within_corr_block"]) == set(c["partial_on_log_length"])
    assert c["spearman_within_corr_block"]["tok_entropy"]["est"] != c["features_vs_honest"]["tok_entropy"]["rho"]
    # the verdict carries both decisions
    v = S.verdict({**H.load_json(fx["out"]), "harness": {}, "signs": {}})
    assert v["attempts"] == base["decision"] and v["attempts_probe_only"] == po["decision"]
    # 'all' again keeps the probe reading of these features; 'both' recomputes both
    S.stage_attempts(_args(fx, "all"))
    assert H.load_json(fx["out"])["attempts"]["probe_only"] == po
    S.stage_attempts(_args(fx, "both"))
    at = H.load_json(fx["out"])["attempts"]
    assert at["probe_only"]["designs"] == po["designs"] and at["designs"] == base["designs"]
    # other features: the stored probe reading is dropped, never carried over
    pd.DataFrame([{**r, "rubric_a": 0.0} for r in fx["rows"]]).to_parquet(
        os.path.join(fx["work"], "features.parquet"), index=False)
    S.stage_attempts(_args(fx, "all"))
    assert "probe_only" not in H.load_json(fx["out"])["attempts"]


def _write_probe_ids(root, ids):
    """kaggle/strong_probe/strong_probe.py under root with only PROBE_IDS, as the notebook spells it."""
    d = os.path.join(root, "kaggle", "strong_probe")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "strong_probe.py"), "w") as fh:
        fh.write('import os\n#: the probe items\nPROBE_IDS = frozenset("""\n' + " ".join(sorted(ids))
                 + '\n""".split())\nOTHER = "x y"\n')


def test_the_probe_flag_is_checked_against_the_notebooks_probe_ids(tmp_path, monkeypatch):
    fx = _probe_fixture(tmp_path, monkeypatch, n=24)
    _write_probe_ids(str(tmp_path), fx["probe"])
    assert S.strong_probe_ids() == frozenset(fx["probe"])
    S.stage_attempts(_args(fx, "probe"))
    chk = H.load_json(fx["out"])["attempts"]["probe_only"]["probe_ids_check"]
    assert chk["ok"] and chk["probe_ids"] == 16 and chk["listed_in_flagged_units"] == 16
    assert chk["units_checked"] == 24 and chk["listed_not_attempted"] == 0
    # a flag that is not PROBE_IDS membership stops the stage, and leaves the stored reading alone
    before = H.load_json(fx["out"])
    _write_probe_ids(str(tmp_path), sorted(fx["probe"])[1:])
    with pytest.raises(SystemExit):
        S.stage_attempts(_args(fx, "both"))
    assert H.load_json(fx["out"]) == before


def test_strong_probe_ids_read_the_notebook_without_importing_it(tmp_path):
    ids = S.strong_probe_ids()
    assert ids is not None and len(ids) == 160                                 # the attempt probe's items
    assert all(len(i) == 16 and int(i, 16) >= 0 for i in ids)
    assert S.strong_probe_ids(str(tmp_path / "missing.py")) is None
    (tmp_path / "x.py").write_text("PROBE = 1\n")
    assert S.strong_probe_ids(str(tmp_path / "x.py")) is None


def test_a_probe_reading_needs_the_flag(tmp_path, monkeypatch):
    fx = _probe_fixture(tmp_path, monkeypatch, n=24)
    os.remove(os.path.join(fx["kaggle"], S.UNITS_DETAIL))
    with pytest.raises(SystemExit):
        S.stage_attempts(_args(fx, "probe"))
    S.stage_attempts(_args(fx, "both"))                     # the default reads every unit and skips the probe
    at = H.load_json(fx["out"])["attempts"]
    assert at["designs"]["cot"]["n_items"] == 24 and "probe_only" not in at


def test_run_facts_from_the_export_and_the_root_manifest(tmp_path):
    kdir = tmp_path / "kaggle"
    (kdir / "_detail").mkdir(parents=True)
    t0 = 1_790_000_000.0
    man = {"slug": "M", "wall_s": 9000.0, "notebook_digest": "d" * 64, "created": "x", "sessions": 1}
    json.dump(man, open(kdir / "manifest.json", "w"))
    json.dump({"attempts": {"units": 10}}, open(kdir / "_summary.json", "w"))
    # rubric rows stamped one by one from t0 + 600 to t0 + 3600; 2,000 prompt tokens each
    pd.DataFrame({"t": np.linspace(t0 + 600, t0 + 3600, 50), "prompt_tokens": 2000, "gen_tokens": 10}).to_parquet(
        kdir / "_detail" / "rubric_units.parquet", index=False)
    # attempts: 10 texts in shards of 5, 500 s apart; the first 7 are probe texts
    ts = [t0 + 3600 + 480] * 5 + [t0 + 3600 + 980] * 5
    write_units(str(kdir), [{"unit": f"u{k}", "benchmark": "matharena", "item_ids": [f"i{k}"] * (1 + (k == 0)),
                             "probe": k < 7, "t": ts[k]} for k in range(10)])
    pd.DataFrame({"n_tokens": [4096] * 38 + [1000, 1000], "capped": [True] * 38 + [False] * 2,
                  "closed": [False] * 38 + [True] * 2, "forced": [True] * 38 + [False] * 2}).to_parquet(
        kdir / "_detail" / "attempt_samples.parquet", index=False)
    raw = tmp_path / "raw"
    root = {"script_sha256": "d" * 64, "wall_s_total": 9000.0, "version": "k1.2",
            "sessions": [{"run": "r", "t0": t0, "wall_s": 9000.0, "status": {"attempts": "deadline"},
                          "stats": {"rubric": {"secs": 2900.0, "prompt_tok_s": 34.0},
                                    "attempts": {"secs": 1480.0, "gen_tok_s": 100.0}}, "files": {"at_end": 9}}],
            "plan": {"M": {"rates_tok_s": {"prefill": 1200}, "rubric": {"hours_est": 0.4, "prompt_tokens": 100000},
                           "attempts": {"units": 30, "probe_units": 7}}},
            "progress": {"M": {"attempts": {"done": 10, "units": 30}}}}
    (raw / "run-1" / "strong_probe").mkdir(parents=True)
    json.dump(root, open(raw / "run-1" / "strong_probe" / "manifest.json", "w"))
    path = S.find_root_manifest(man, str(raw))
    assert path == str(raw / "run-1" / "strong_probe" / "manifest.json")
    assert S.find_root_manifest({**man, "wall_s": 1.0}, str(raw)) is None          # another session's export
    (raw / "run-2" / "strong_probe").mkdir(parents=True)
    json.dump(root, open(raw / "run-2" / "strong_probe" / "manifest.json", "w"))
    assert S.find_root_manifest(man, str(raw)) is None                              # ambiguous
    f = S.run_facts(str(kdir), path)
    assert f["logs"] == []                                                      # no log saved beside the copy
    tl, der = f["timeline"], f["derived"]
    assert tl["attempt_units"] == {"probe": 7, "rest": 3} and tl["attempt_item_ids"] == {"probe": 8, "rest": 3}
    assert tl["attempt_shards"] == 2 and tl["units_per_shard"] == 5.0 and tl["shard_s"]["median"] == 500.0
    assert tl["first_shard_after_rubric_s"] == 480.0 and tl["probe_done_after_rubric_h"] == round(980 / 3600, 3)
    assert tl["rubric_prompt_tokens"] == 100000 and tl["attempt_gen_tokens"] == 38 * 4096 + 2000
    assert tl["attempt_gen_tok_s"] == round((38 * 4096 + 2000) / 980, 1)
    assert tl["attempt_capped_share"] == 0.95 and tl["attempt_closed_share"] == 0.05
    assert der["rubric_hours_over_plan"] == round(2900 / 3600 / 0.4, 2)
    assert der["before_rubric_s"] == 700.0                    # the last rubric write less its seconds, from t0
    assert der["attempt_units_left"] == 20 and der["attempt_hours_left_at_median_shard"] == round(4 * 500 / 3600, 2)
    assert f["sessions"][0]["t0_utc"] == S._utc(t0) and f["plan"]["attempts"]["probe_units"] == 7
    # without the root manifest: the export's own facts only
    g = S.run_facts(str(kdir))
    assert g["root_manifest"] is None and g["timeline"] == tl and "derived" not in g
    # a log saved beside the Output copy is recorded with its digest and what it shows
    text = ("RuntimeError: ... prefix_prefill ... Unsupported conversion from f16 to f16\n"
            "retrying with --no-prefix-caching\n")
    (raw / "run-1" / "session.log").write_text(text)
    lg = S.run_facts(str(kdir), path)["logs"]
    assert len(lg) == 1 and lg[0]["path"].endswith(os.path.join("run-1", "session.log"))
    assert lg[0]["sha256"] == hashlib.sha256(text.encode()).hexdigest() and lg[0]["bytes"] == len(text.encode())
    assert lg[0]["mentions"] == {"prefix_prefill": True, "f16_conversion": True, "no_prefix_caching": True}


# --- the entropy job (commit D) ------------------------------------------------------------------------

#: entropy/entropy.parquet's dtypes as kaggle/strong_probe/strong_probe.py writes them (ENTROPY_COLS)
ENT_DTYPES = {"ent_first256": "float64", "ent_first1024": "float64", "lp_first256": "float64",
              "lp_first1024": "float64", "ent_n_tokens": "int64", "ent_closed": "bool", "ent_degenerate": "bool",
              "ent_prompt_tokens": "int64", "ent_task_tokens": "int64", "ent_truncated": "bool"}
ENT_DIAG = ("ent_degenerate", "ent_prompt_tokens", "ent_task_tokens", "ent_truncated")


def entropy_frame(items, fn=None, benches=S.PARENTS):
    """The entropy table as the kit writes it: the key columns, then its columns in its dtypes; fn(b, iid) ->
    {column: value} overrides the defaults."""
    rows = []
    for b in benches:
        for k, (iid, it) in enumerate(items[b].items()):
            r = {"benchmark": b, "item_id": iid, "content_sha256": S.content_sha256(it), "ent_first256": 2.0 + 0.01 * k,
                 "ent_first1024": 2.5 + 0.01 * k, "lp_first256": -0.7, "lp_first1024": -0.8, "ent_n_tokens": 1024,
                 "ent_closed": False, "ent_degenerate": False, "ent_prompt_tokens": 400, "ent_task_tokens": 380,
                 "ent_truncated": False}
            r.update(fn(b, iid) if fn else {})
            rows.append(r)
    return pd.DataFrame(rows).astype(ENT_DTYPES)


def entropy_kind(**over):
    """kinds.entropy as the kit's export manifest writes it (the fields this script reads)."""
    k = {"version": S.ENTROPY_VERSION, "cfg": "e21faa7f3d0bf929", "cfg_hash": "e21faa7f3d0bf929",
         "primary": "ent_first1024", "signs": {**S.ENTROPY_FEATURES, **{c: 0 for c in S.ENTROPY_COLUMNS
                                                                         if c not in S.ENTROPY_FEATURES}},
         "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
         "n": 1, "max_new_tokens": 1024, "logprobs": "raw",
         "token_stats": {"logprobs": "raw", "sources": {"raw_full_vocab": 1},
                         "recorder_check": {"status": "ok", "max_abs_gap": 0.0}}}
    k.update(over)
    return k


def write_export(root, shards, kinds, **extra):
    """An export directory: {relative path: DataFrame} as parquet, each listed in manifest.json with its rows and
    sha256, and the given kinds."""
    os.makedirs(root, exist_ok=True)
    listed = []
    for rel, df in shards.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_parquet(p, index=False)
        listed.append({"path": rel, "rows": len(df), "sha256": S.file_sha256(p)})
    man = {"schema_version": S.SCHEMA_VERSION, "model": {"repo": "org/model", "revision": "abc"},
           "hash": S.HASH_DEF, "kinds": kinds, "shards": listed, "logprobs": "raw", **extra}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    return man


def commit_d(first, root, ent, kind=None):
    """Commit D with the first commit's Output attached: its rubric and attempt tables byte for byte, the entropy
    table beside them, kinds.entropy added and the session fields changed."""
    shutil.copytree(first, root)
    with open(os.path.join(root, "manifest.json")) as fh:
        man = json.load(fh)
    p = os.path.join(root, "entropy", "entropy.parquet")
    os.makedirs(os.path.dirname(p))
    ent.to_parquet(p, index=False)
    man["shards"].append({"path": "entropy/entropy.parquet", "rows": len(ent), "sha256": S.file_sha256(p)})
    man["kinds"]["entropy"] = kind or entropy_kind()
    man.update(created="commit D", wall_s=2.0, sessions=2)
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(man, fh)


def test_the_entropy_table_passes_the_schema_as_its_own_kind(tmp_path):
    items, _ = make_items(8)
    ent = entropy_frame(items)
    write_export(str(tmp_path / "e"), {"entropy/entropy.parquet": ent}, {"entropy": entropy_kind()})
    rep = S.check_schema(str(tmp_path / "e"))
    assert rep["ok"] and not rep["warnings"], (rep["errors"], rep["warnings"])
    info = rep["shards"]["entropy/entropy.parquet"]
    assert info["kind"] == "entropy" and info["features"] == list(S.ENTROPY_COLUMNS) == list(ENT_DTYPES)
    # beside the rubric (whose own prompt_tokens it must not collide with) and the attempts
    rub = pd.DataFrame(rubric_rows(items, lambda b, i: {"rubric_a": 1.0, "prompt_tokens": 10.0}))
    att = pd.DataFrame([r for iid, it in items["matharena"].items() for r in attempt_rows("matharena", iid, it, ["1"])])
    write_export(str(tmp_path / "d"), {"rubric/rubric.parquet": rub, "attempts/attempts.parquet": att,
                                       "entropy/entropy.parquet": ent},
                 {"rubric": {"prompt": "p"}, "attempts": {"prompt": "q"}, "entropy": entropy_kind()})
    rep = S.check_schema(str(tmp_path / "d"))
    assert rep["ok"] and not rep["warnings"], (rep["errors"], rep["warnings"])
    man, frames = S.load_shards(str(tmp_path / "d"))
    assert [(rel.split(os.sep)[0], kind) for rel, kind, _ in frames] == [("attempts", "attempts"),
                                                                         ("entropy", "entropy"), ("rubric", "items")]
    # never merged into the rubric's and the attempts' features
    assert set(S.item_table(frames).columns) == {"benchmark", "item_id", "content_sha256", "rubric_a", "prompt_tokens"}
    et, deg = S.entropy_table(frames)
    assert set(S.ENTROPY_COLUMNS) <= set(et.columns) and len(et) == 32 and deg == {}
    assert not set(S.ENTROPY_COLUMNS) & set(S.combine(S.item_table(frames), S.attempt_table(frames))[0].columns)
    assert S.SCHEMA["entropy_shard"]["required"] == list(S.ENTROPY_FEATURES)


def test_the_schema_flags_entropy_problems(tmp_path):
    items, _ = make_items(6)
    n = iter(range(100))

    def check(ent=None, kind="default", rubric=None):
        d = str(tmp_path / f"x{next(n)}")
        shards = {"entropy/entropy.parquet": entropy_frame(items) if ent is None else ent}
        if rubric is not None:
            shards["rubric/rubric.parquet"] = rubric
        write_export(d, shards, {} if kind is None else {"entropy": entropy_kind() if kind == "default" else kind})
        return S.check_schema(d)

    r = check(entropy_frame(items).drop(columns="ent_first1024"))
    assert not r["ok"] and any("entropy shard lacks ['ent_first1024']" in e for e in r["errors"])
    r = check(entropy_frame(items).astype({"ent_closed": float}).assign(ent_closed=0.5))
    assert not r["ok"] and any("ent_closed is not bool" in e for e in r["errors"])
    r = check(entropy_frame(items).assign(ent_n_tokens=1.5))
    assert not r["ok"] and any("ent_n_tokens is not int" in e for e in r["errors"])
    r = check(entropy_frame(items).assign(mystery=1.0))
    assert r["ok"] and any("entropy columns not read: ['mystery']" in w for w in r["warnings"])
    r = check(kind=None)
    assert r["ok"] and any("no kinds.entropy" in w for w in r["warnings"])
    failed = {"logprobs": "raw", "recorder_check": {"status": "FAILED", "max_abs_gap": 0.5}}
    for kind, what in ((entropy_kind(signs={"ent_first1024": -1}), "differ from ENTROPY_SIGNS"),
                       (entropy_kind(primary="lp_first1024"), "the declared primary is 'ent_first1024'"),
                       (entropy_kind(version="e9.9"), "the primary cannot be kept"),
                       (entropy_kind(token_stats=failed), "the recorder check failed"),
                       (entropy_kind(logprobs="raw_topk", token_stats={"logprobs": "raw_topk"}),
                        "entropy features excluded: token statistics 'raw_topk'")):
        r = check(kind=kind)
        assert r["ok"] and any(what in w for w in r["warnings"]), (what, r["warnings"])
    # a listed row count or sha256 that the file does not match
    d = str(tmp_path / "rows")
    man = write_export(d, {"entropy/entropy.parquet": entropy_frame(items)}, {"entropy": entropy_kind()})
    man["shards"][0].update(rows=1, sha256="0" * 64)
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    errs = "\n".join(S.check_schema(d)["errors"])
    assert "rows, manifest says 1" in errs and "sha256 differs from the manifest" in errs
    for listing in ([], [{"path": "entropy/entropy.parquet", "rows": 24}]):      # unlisted, or without its sha256
        man["shards"] = listing
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
        r = S.check_schema(d)
        assert not r["ok"] and any("must be listed in the manifest's shards" in e for e in r["errors"])
    # entropy names outside entropy/ are read as the rubric's kind, and said so
    r = check(rubric=pd.DataFrame(rubric_rows(items, lambda b, i: {"rubric_a": 1.0, "ent_first1024": 2.0})))
    assert any("entropy column names outside entropy/" in w for w in r["warnings"])


def test_entropy_semantics_and_registry():
    sem = S.entropy_semantics({"kinds": {"entropy": entropy_kind()}})
    assert sem["readable"] and sem["primary_eligible"] and sem["logprobs"] == "raw" and not sem["sign_disagreements"]
    none = S.entropy_semantics({})
    assert not none["readable"] and "no kinds.entropy" in none["why"] and not none["primary_eligible"]
    failed = entropy_kind(token_stats={"logprobs": "raw", "recorder_check": {"status": "FAILED"}})
    assert not S.entropy_semantics({"kinds": {"entropy": failed}})["readable"]
    for lp in ("raw_topk", "processed_topk", "mixed"):
        s = S.entropy_semantics({"kinds": {"entropy": entropy_kind(logprobs=lp, token_stats={"logprobs": lp})}})
        assert not s["readable"] and lp in s["why"]
    na = entropy_kind(token_stats={"logprobs": "raw", "recorder_check": {"status": "n/a"}})
    assert S.entropy_semantics({"kinds": {"entropy": na}})["readable"]              # unchecked is not failed
    other = S.entropy_semantics({"kinds": {"entropy": entropy_kind(version="e2.0")}})
    assert other["readable"] and not other["primary_eligible"]
    d = S.entropy_semantics({"kinds": {"entropy": entropy_kind(signs={"lp_first256": 1, "ent_first256": "x"})}})
    assert d["sign_disagreements"] == {"lp_first256": 1, "ent_first256": "x"}
    # the declared signs: entropy + (harder), log-prob -, generated length +, thinking closed -
    assert S.ENTROPY_SIGNS == {"ent_first256": 1, "ent_first1024": 1, "lp_first256": -1, "lp_first1024": -1,
                               "ent_n_tokens": 1, "ent_closed": -1}
    assert S.ENTROPY_PRIMARY == S.ENTROPY_RULE["primary"] == "ent_first1024"
    assert sorted(S.ENTROPY_RULE["exploratory"]) == sorted(set(S.ENTROPY_SIGNS) - {"ent_first1024"})
    j = pd.DataFrame({"benchmark": ["matharena"] * 3, "item_id": list("abc"), "key": "k", "key_official": "k",
                      "text_key": "t", "content_sha256": "h", "source": "direct", "prompt_sha": "p",
                      **{c: [1.0, 2.0, 3.0] for c in S.ENTROPY_COLUMNS}})
    reg = S.entropy_registry(j, sem)
    assert set(reg) == set(S.ENTROPY_COLUMNS)
    assert {c: v["sign"] for c, v in reg.items()} == {**S.ENTROPY_SIGNS, **{c: 0 for c in ENT_DIAG}}
    assert reg["ent_first1024"]["primary"] and reg["ent_first1024"]["role"] == "primary"
    assert sorted(c for c, v in reg.items() if v["role"] == "exploratory") == sorted(S.ENTROPY_RULE["exploratory"])
    assert {c for c, v in reg.items() if v["usable"]} == set(S.ENTROPY_SIGNS)
    m = S.oriented_maps(j, reg)
    assert m["lp_first1024"]["c"] == -3.0 and m["ent_closed"]["a"] == -1.0 and m["ent_first1024"]["c"] == 3.0
    assert m["ent_prompt_tokens"]["c"] == 3.0                                        # a diagnostic: raw
    assert set(S.entropy_harness_maps(j, reg)) == set(S.ENTROPY_SIGNS)
    assert set(S.entropy_harness_maps(j, reg, ["ent_first1024"])) == {"ent_first1024"}
    bad = S.entropy_registry(j, S.entropy_semantics({"kinds": {"entropy": failed}}))
    assert not any(v["usable"] for v in bad.values()) and "recorder" in bad["ent_first1024"]["excluded"]
    assert S.entropy_harness_maps(j, bad) == {}
    # the rubric-and-attempt registry has no entropy kind
    assert S.head_specs(S.registry(j.drop(columns=list(S.ENTROPY_COLUMNS)))) == {}


#: the entropy job's decision as fixed on 2026-09-28 before commit D's output was read (README, "Коммит D", quotes
#: it): a change to ENTROPY_RULE, ENTROPY_SIGNS, ENTROPY_PRIMARY, ENTROPY_CFG or the gate fails here first
ENTROPY_RULE_DIGEST = "72cde00805ff3c87d7596581426dc0296316ad0734b24d9e85015d304cfe2d3a"
ENTROPY_DECLARED_DIGEST = "390910ea499789c8dab96c40bb3efc3e10a5438ede50ed9649af617a767f8ccb"


def test_the_entropy_rule_is_pinned():
    assert S.entropy_rule_digest() == ENTROPY_RULE_DIGEST == S.entropy_rule_record()["digest"]
    assert F.digest(H._jsonable({"primary": S.ENTROPY_PRIMARY, "signs": S.ENTROPY_SIGNS})) == ENTROPY_DECLARED_DIGEST
    assert S.ENTROPY_PRIMARY == S.ENTROPY_RULE["primary"] == "ent_first1024"
    assert S.ENTROPY_SIGNS == S.ENTROPY_RULE["signs"] == {"ent_first256": 1, "ent_first1024": 1, "lp_first256": -1,
                                                          "lp_first1024": -1, "ent_n_tokens": 1, "ent_closed": -1}
    assert S.ENTROPY_RULE["config"] == {**S.ENTROPY_RULE["config"], "version": "e1.0", "cfg": "e21faa7f3d0bf929"}
    assert S.ENTROPY_CFG == "e21faa7f3d0bf929" and S.ENTROPY_VERSION == "e1.0"
    # the config is the kit's with every default, commit D's ARGS
    from tests import test_kaggle_probe as KP
    SP = KP.SP
    d = SP.parse_args(["run", "--jobs", "entropy", "--no-prefix-caching"])
    assert SP.digest(SP.job_cfg(d, "entropy")) == S.ENTROPY_CFG and SP.ENTROPY_VERSION == S.ENTROPY_VERSION
    # the digest is of the content: a stored rule is checked against itself
    rec = S.entropy_rule_record()
    assert S.entropy_rule_digest(json.loads(json.dumps(H._jsonable(rec)))) == ENTROPY_RULE_DIGEST
    assert S.entropy_rule_digest({**rec, "primary": "ent_first256"}) != ENTROPY_RULE_DIGEST


def test_the_entropy_table_joins_by_content_hash_and_reports_coverage():
    items, _ = make_items(10)

    def fn(b, iid):
        k = int(iid[3:])
        out = {"ent_first1024": 2.0 + k}
        if b == "multi_swebench" and k < 2:
            out.update(ent_degenerate=True, ent_first1024=0.01)        # fp16 overflow's '!!!!'
        if b == "real_webagents" and k == 0:
            out.update(ent_first1024=float("nan"), ent_n_tokens=0)     # an empty sample
        return out

    ent = entropy_frame(items, fn)
    ent.loc[ent.item_id == "res0003", "content_sha256"] = "0" * 64       # a text that is not the local one
    frames = [("entropy/entropy.parquet", "entropy", S._norm(ent))]
    table, deg = S.entropy_table(frames)
    assert deg == {"multi_swebench": 2}
    t = table.set_index("item_id")
    assert np.isnan(t.loc["mul0000", "ent_first1024"]) and np.isnan(t.loc["mul0001", "ent_closed"])
    assert np.isnan(t.loc["mul0000", "ent_n_tokens"]) and t.loc["mul0000", "ent_degenerate"] == 1.0
    assert t.loc["mul0000", "ent_prompt_tokens"] == 400.0 and t.loc["mat0003", "ent_first1024"] == 5.0
    joined, rep = S.join(table, items)
    r = rep["per_benchmark"]
    assert r["researchcodebench"]["hash_mismatch"] == 1 and r["researchcodebench"]["covered_items"] == 9
    assert all(r[b]["covered_items"] == 10 for b in ("matharena", "multi_swebench", "real_webagents"))
    st = S.entropy_per_benchmark(joined, deg)
    assert st["multi_swebench"]["primary_finite"] == 8 and st["multi_swebench"]["degenerate_excluded"] == 2
    assert st["multi_swebench"]["rate_ent_degenerate"] == pytest.approx(0.2)
    assert st["real_webagents"]["primary_finite"] == 9 and st["matharena"]["mean_ent_first1024"] == pytest.approx(6.5)
    assert st["matharena"]["mean_ent_n_tokens"] == 1024 and st["matharena"]["rate_ent_closed"] == 0


def test_response_coverage_reads_each_benchmark_once_for_both_tables(monkeypatch):
    from paiec import data as D
    from paiec.evaluator import Pair, Response
    items, _ = make_items(6)
    keys = S.item_keys(items)
    calls = []

    def fake(bs):
        calls.append(tuple(bs))
        return [Pair({}, "s", bs[0], [Response(i, it, 1) for i, it in items[bs[0]].items()])]

    monkeypatch.setattr(D, "load_pairs", fake)
    ids = [(b, i) for b in S.PARENTS for i in items[b]]
    jm = pd.DataFrame({"benchmark": [b for b, _ in ids], "item_id": [i for _, i in ids],
                       "rubric_a": 1.0, "att_cot_k": [4.0 if i.endswith(("0", "1")) else np.nan for _, i in ids]})
    je = pd.DataFrame({"benchmark": [b for b, _ in ids[:12]], "item_id": [i for _, i in ids[:12]],
                       "ent_first1024": [np.nan, np.nan] + [1.0] * 10})
    one = S.response_coverage(jm, list(items), keys)
    assert one["matharena"] == {"pairs": 1, "responses": 6, "covered_share": 1.0, "covered_share_rubric": 1.0,
                                "covered_share_attempts": round(2 / 6, 4), "item_key_mismatches": 0}
    calls.clear()
    both = S.responses_coverage({"main": (jm, {"rubric": ["rubric_a"], "attempts": ["att_cot_k"]}),
                                 "entropy": (je, {"entropy": ["ent_first1024"]})}, list(items), keys)
    assert both["main"] == one and len(calls) == 4                   # one read per benchmark
    assert both["entropy"]["matharena"]["covered_share"] == 1.0
    assert both["entropy"]["matharena"]["covered_share_entropy"] == round(4 / 6, 4)
    assert both["entropy"]["real_webagents"]["covered_share"] == 0.0


# --- ingesting commit D --------------------------------------------------------------------------------

def write_data(root, items, subjects=("s1", "s2")):
    """data/<benchmark>/{items,response,subjects}.parquet in measurement-db's layout, every subject answering
    every item (what local_items and paiec.data.load_pairs read)."""
    from paiec import data as D
    for b, d in items.items():
        p = os.path.join(root, b)
        os.makedirs(p, exist_ok=True)
        pd.DataFrame({"item_id": list(d), "benchmark_id": b, "content": [it["item_content"] for it in d.values()],
                      "item_features": [it["item_features"] for it in d.values()],
                      "grading_criterion": [json.dumps({"reference_answer": "1"})] * len(d)}).to_parquet(
            os.path.join(p, "items.parquet"), index=False)
        pd.DataFrame([{"subject_id": s, "benchmark_id": b, "item_id": i, "response": float(k % 2)}
                      for s in subjects for k, i in enumerate(d)]).to_parquet(os.path.join(p, "response.parquet"),
                                                                              index=False)
        pd.DataFrame({"subject_id": list(subjects), **{f: "x" for f in D.SUBJECT_FIELDS}}).to_parquet(
            os.path.join(p, "subjects.parquet"), index=False)


def _ingest_fixture(tmp_path, monkeypatch, n=80):
    """Public items on disk (paiec.data.DATA_DIR) and the first commit's export A: a rubric on every parent item,
    attempts on 20 matharena items."""
    from paiec import data as D
    items, diff = make_items(n, seed=31)
    write_data(str(tmp_path / "data"), items)
    monkeypatch.setattr(D, "DATA_DIR", str(tmp_path / "data"))
    rub = pd.DataFrame(rubric_rows(items, lambda b, i: {"rubric_a": diff[(b, i)], "rubric_b": 1.0}))
    att = pd.DataFrame([r for iid, it in list(items["matharena"].items())[:20]
                        for r in attempt_rows("matharena", iid, it, ["1", "2"])])
    a = tmp_path / "A"
    write_export(str(a), {"rubric/rubric.parquet": rub, "attempts/attempts.parquet": att},
                 {"rubric": {"prompt": "p", "signs": {"rubric_b": -1}},
                  "attempts": {"prompt": "q", "sampling": {"temperature": 0.6, "presence_penalty": 1.5}}},
                 created="first")
    return items, diff, a


def _ing(tmp_path, *dirs):
    return SimpleNamespace(kaggle=[str(d) for d in dirs], work=str(tmp_path / "work"),
                           out=str(tmp_path / "out.json"), force=False)


def _main_part(state):
    return {k: v for k, v in state.items() if k != S.ENTROPY_KIND}


def test_commit_d_leaves_every_stored_result_of_the_rubric_and_attempts(tmp_path, monkeypatch):
    items, diff, a = _ingest_fixture(tmp_path, monkeypatch)
    args = _ing(tmp_path, a)
    S.stage_ingest(args)
    st = H.load_json(args.out)
    assert S.ENTROPY_KIND not in st and st["ingest"]["per_benchmark"]["matharena"]["covered_items"] == 80
    # the stored results of the first commit (their contents do not matter here), and the heads' predictions
    dig = st["ingest"]["features_digest"]
    st.update(signs={"features_digest": dig, "features": {"rubric_a": {"n": 1}}}, heads={"rubric_ridge": {"r": 0.19}},
              harness={"_meta": {}, "rubric_a": {"lines": {}}}, reference={"r": {}}, attempts={"decision": "GO"},
              verdict={"call": "NULL"}, run={"sessions": [1]})
    H.save_json(args.out, st)
    oof = os.path.join(args.work, "oof.json")
    H.save_json(oof, {"features_digest": dig, "heads": {"rubric_ridge": {"mat0000": 0.1}}})
    before = H.load_json(args.out)
    with open(args.out) as fh:
        before_text = fh.read()
    with open(os.path.join(args.work, "features.parquet"), "rb") as fh:
        feats = fh.read()
    ent = entropy_frame(items, lambda b, i: {"ent_first1024": 2.0 + diff[(b, i)]})
    d = tmp_path / "D"
    commit_d(str(a), str(d), ent)
    chk = S.check_schema(str(d))
    assert chk["ok"] and not chk["warnings"], (chk["errors"], chk["warnings"])
    S.stage_ingest(_ing(tmp_path, d))
    after = H.load_json(args.out)
    assert _main_part(after) == before and list(after)[:-1] == list(before)
    with open(args.out) as fh:          # the rubric's and the attempts' sections are the same text, entropy is added
        assert fh.read().startswith(before_text[:before_text.rindex("\n}")])
    with open(os.path.join(args.work, "features.parquet"), "rb") as fh:
        assert fh.read() == feats
    assert H.load_json(oof)["features_digest"] == dig
    e = after[S.ENTROPY_KIND]
    assert e["rule"] == json.loads(json.dumps(H._jsonable(S.entropy_rule_record())))
    assert e["rule"]["digest"] == S.entropy_rule_digest(e["rule"]) == e["ingest"]["rule_digest"]
    ing = e["ingest"]
    assert ing["main_features"]["status"].startswith("byte-identical")
    assert ing["main_features"]["features_digest"] == dig
    for b, r in ing["per_benchmark"].items():
        assert r["rows"] == r["hash_ok"] == r["covered_items"] == 80, (b, r)
        assert ing["responses"][b]["covered_share_entropy"] == 1.0 and ing["responses"][b]["responses"] == 160
    assert ing["semantics"]["readable"] and ing["semantics"]["primary_eligible"]
    assert ing["features_digest"] == S.entropy_digest(args.work) and e["meta"]["kaggle_dir"] == str(d)
    je = S.load_entropy(args.work)
    assert list(je.columns[:8]) == list(S.JOIN_COLS) + ["prompt_sha"] and len(je) == 320
    assert je.set_index("item_id").loc["mat0005", "ent_first1024"] == pytest.approx(2.0 + diff[("matharena",
                                                                                                 "mat0005")])
    # the same export again: nothing of the entropy job's is dropped either
    e["signs"], e["consistency"] = {"kept": True}, {"kept": True}
    H.save_json(args.out, after)
    S.stage_ingest(_ing(tmp_path, d))
    again = H.load_json(args.out)
    assert again[S.ENTROPY_KIND]["signs"] == {"kept": True} and _main_part(again) == before
    # another entropy table: its downstream results go, the rest stays
    d2 = tmp_path / "D2"
    commit_d(str(a), str(d2), ent.assign(ent_first1024=ent.ent_first1024 + 1.0))
    S.stage_ingest(_ing(tmp_path, d2))
    st2 = H.load_json(args.out)
    assert not {"signs", "consistency"} & set(st2[S.ENTROPY_KIND]) and _main_part(st2) == before
    assert st2[S.ENTROPY_KIND]["ingest"]["features_digest"] != ing["features_digest"]


def test_ingest_keeps_the_entropy_rule_traceable(tmp_path, monkeypatch):
    """entropy.rule carries its digest, entropy.rule_first the first ingest's, entropy.history one entry per
    ingest; a changed ENTROPY_RULE stops ingest before it writes anything, unless --accept-rule-change, which keeps
    the old rule in entropy.rule_history and drops what was read under it."""
    items, diff, a = _ingest_fixture(tmp_path, monkeypatch, n=24)
    d = tmp_path / "D"
    commit_d(str(a), str(d), entropy_frame(items, lambda b, i: {"ent_first1024": 2.0 + diff[(b, i)]}))
    args = _ing(tmp_path, d)
    S.stage_ingest(args)
    e = H.load_json(args.out)[S.ENTROPY_KIND]
    dig = S.entropy_rule_digest()
    assert e["rule"]["digest"] == e["ingest"]["rule_digest"] == e["rule_first"]["digest"] == dig
    assert e["rule"] == json.loads(json.dumps(H._jsonable(S.entropy_rule_record())))
    comp = e["ingest"]["completeness"]
    assert comp["complete"] and comp["units_expected"] == comp["units_covered"] == 96
    assert all(v["units_expected"] == v["units_covered"] == 24 and v["covered_share"] == 1.0
               for v in comp["per_parent"].values())
    h = e["history"]
    assert len(h) == 1 and h[0]["stage"] == "ingest" and h[0]["cfg"] == S.ENTROPY_CFG and h[0]["complete"]
    assert h[0]["rule_digest"] == dig and h[0]["features_digest"] == S.entropy_digest(args.work)
    # results read under the rule
    st = H.load_json(args.out)
    st[S.ENTROPY_KIND].update(signs={"s": 1}, harness={"h": 1}, verdict={"call": "KEEP"})
    H.save_json(args.out, st)
    before, table = H.load_json(args.out), S.entropy_digest(args.work)
    with open(os.path.join(args.work, "features.parquet"), "rb") as fh:
        feats = fh.read()
    # a changed declaration (the primary): ingest refuses before it reads or writes anything
    monkeypatch.setitem(S.ENTROPY_RULE, "primary", "ent_first256")
    with pytest.raises(SystemExit, match="differs from the rule"):
        S.stage_ingest(args)
    assert H.load_json(args.out) == before and S.entropy_digest(args.work) == table
    with open(os.path.join(args.work, "features.parquet"), "rb") as fh:
        assert fh.read() == feats
    # --accept-rule-change: recorded, and what was read under the old rule goes; the first rule stays
    S.stage_ingest(SimpleNamespace(**vars(args), accept_rule_change=True))
    e2 = H.load_json(args.out)[S.ENTROPY_KIND]
    rh = e2["rule_history"]
    assert [r["digest"] for r in rh] == [dig] and rh[0]["fields"] == ["primary"] and rh[0]["replaced_by"] != dig
    assert rh[0]["rule"]["primary"] == "ent_first1024" and e2["rule"]["primary"] == "ent_first256"
    assert e2["rule"]["digest"] == S.entropy_rule_digest() != dig and e2["rule_first"]["digest"] == dig
    assert not {"signs", "harness", "verdict"} & set(e2)
    assert [x["stage"] for x in e2["history"]] == ["ingest", "ingest"] and e2["history"][1]["rule_changed"]
    assert e2["history"][1]["dropped"] == ["signs", "harness", "verdict"]
    # changing it back is a change too; a stored rule edited by hand no longer matches its digest
    monkeypatch.setitem(S.ENTROPY_RULE, "primary", "ent_first1024")
    with pytest.raises(SystemExit, match="differs from the rule"):
        S.stage_ingest(args)
    st = H.load_json(args.out)
    st[S.ENTROPY_KIND]["rule"] = {**S.entropy_rule_record(), "keep": "anything"}
    H.save_json(args.out, st)
    with pytest.raises(SystemExit, match="no longer matches its own digest"):
        S.stage_ingest(args)
    # another job config (another seed, model or cut): read, but the primary cannot be kept, and it is on record
    d2 = tmp_path / "D2"
    commit_d(str(a), str(d2), entropy_frame(items), kind=entropy_kind(cfg="ffffffffffffffff"))
    chk = S.check_schema(str(d2))
    assert chk["ok"] and any("job config 'ffffffffffffffff'" in w for w in chk["warnings"])
    a2 = SimpleNamespace(kaggle=[str(d2)], work=str(tmp_path / "work2"), out=str(tmp_path / "out2.json"), force=False)
    S.stage_ingest(a2)
    e3 = H.load_json(a2.out)[S.ENTROPY_KIND]
    sem = e3["ingest"]["semantics"]
    assert sem["readable"] and not sem["primary_eligible"] and "job config" in sem["why_not_eligible"]
    assert e3["history"][-1]["cfg"] == "ffffffffffffffff" and not e3["history"][-1]["primary_eligible"]
    # a partial export (a commit stopped with exit 75): not complete, per parent
    d3 = tmp_path / "D3"
    ent = entropy_frame(items)
    commit_d(str(a), str(d3), ent[~ent.item_id.isin([f"mat{k:04d}" for k in range(4)])])
    a3 = SimpleNamespace(kaggle=[str(d3)], work=str(tmp_path / "work3"), out=str(tmp_path / "out3.json"), force=False)
    S.stage_ingest(a3)
    comp = H.load_json(a3.out)[S.ENTROPY_KIND]["ingest"]["completeness"]
    assert not comp["complete"] and comp["per_parent"]["matharena"]["units_covered"] == 20
    assert not comp["per_parent"]["matharena"]["complete"] and comp["per_parent"]["real_webagents"]["complete"]


def test_an_entropy_only_export_keeps_the_features_and_two_exports_merge(tmp_path, monkeypatch):
    items, diff, a = _ingest_fixture(tmp_path, monkeypatch, n=24)
    args = _ing(tmp_path, a)
    S.stage_ingest(args)
    st0 = H.load_json(args.out)
    st0["signs"] = {"kept": 1}
    H.save_json(args.out, st0)
    features = S.features_digest(args.work)
    e = tmp_path / "E"
    write_export(str(e), {"entropy/entropy.parquet": entropy_frame(items)}, {"entropy": entropy_kind()})
    S.stage_ingest(_ing(tmp_path, e))
    st = H.load_json(args.out)
    assert _main_part(st) == st0 and S.features_digest(args.work) == features
    mf = st[S.ENTROPY_KIND]["ingest"]["main_features"]
    assert mf["source"] is None and "kept" in mf["status"] and mf["features_digest"] == st0["ingest"]["features_digest"]
    # both directories: the rubric and the attempts from A (unchanged), the entropy table from E
    chk, src = S.check_exports([str(a), str(e)])
    assert chk["ok"] and chk["sources"] == {"main": str(a), "entropy": str(e)}
    S.stage_ingest(_ing(tmp_path, a, e))
    st = H.load_json(args.out)
    assert _main_part(st) == st0 and st[S.ENTROPY_KIND]["meta"]["kaggle_dir"] == str(e)
    assert st[S.ENTROPY_KIND]["ingest"]["main_features"]["source"] == str(a)
    # commit D beside A carries the same rubric and attempts: read once; a different rubric is an error
    d = tmp_path / "D"
    commit_d(str(a), str(d), entropy_frame(items))
    chk, src = S.check_exports([str(a), str(d)])
    assert chk["ok"] and chk["sources"] == {"main": str(a), "entropy": str(d)} and not chk["warnings"]
    # the same rows in other bytes (Kaggle's pyarrow writes the attached tables anew): read once, with a warning
    db = tmp_path / "D_bytes"
    shutil.copytree(str(d), str(db))
    rp = db / "rubric" / "rubric.parquet"
    pd.read_parquet(rp).to_parquet(rp, index=False, compression="gzip")
    with open(db / "manifest.json") as fh:
        man = json.load(fh)
    for sh in man["shards"]:
        sh["sha256"] = S.file_sha256(str(db / sh["path"]))
    with open(db / "manifest.json", "w") as fh:
        json.dump(man, fh)
    assert S.file_sha256(str(rp)) != S.file_sha256(str(a / "rubric" / "rubric.parquet"))
    chk, src = S.check_exports([str(a), str(db)])
    assert chk["ok"] and chk["sources"] == {"main": str(a), "entropy": str(db)}
    assert any("same rows" in w and "other bytes" in w for w in chk["warnings"])
    a2 = tmp_path / "A2"
    rub2 = pd.DataFrame(rubric_rows(items, lambda b, i: {"rubric_a": diff[(b, i)] + 1.0, "rubric_b": 1.0}))
    write_export(str(a2), {"rubric/rubric.parquet": rub2}, {"rubric": {"prompt": "p"}})
    chk, _ = S.check_exports([str(a), str(a2)])
    assert not chk["ok"] and any(err.startswith("main:") for err in chk["errors"])
    with pytest.raises(SystemExit):
        S.stage_ingest(_ing(tmp_path, a, a2))
    with pytest.raises(SystemExit):
        S.stage_check_schema(_ing(tmp_path, a, a2))
    assert _main_part(H.load_json(args.out)) == st0
    # another rubric: its results go as before, and so does the consistency check that read them
    st = H.load_json(args.out)
    st[S.ENTROPY_KIND]["consistency"] = {"x": 1}
    H.save_json(args.out, st)
    S.stage_ingest(_ing(tmp_path, a2))
    st = H.load_json(args.out)
    assert "signs" not in st and st["ingest"]["features_digest"] != st0["ingest"]["features_digest"]
    assert "consistency" not in st[S.ENTROPY_KIND] and "ingest" in st[S.ENTROPY_KIND]


# --- the entropy job's stages ---------------------------------------------------------------------------

def _entropy_work(tmp_path, monkeypatch, n=40, semantics=None):
    """WORK/entropy.parquet with planted features (ent_first1024 follows the difficulty, lp_first1024 against it,
    lp_first256 noise; matharena's prompts in pairs of identical text), its ingest state, and the stage's
    inputs stubbed."""
    items, diff = make_items(n, seed=41)
    rng = np.random.default_rng(42)
    rows = []
    for b in S.PARENTS:
        for k, (i, it) in enumerate(items[b].items()):
            dd = diff[(b, i)]
            rows.append({"benchmark": b, "item_id": i, "key": f"k{i}", "key_official": "k", "text_key": "t",
                         "content_sha256": "h", "source": "direct",
                         "prompt_sha": f"p{b[:3]}{k // 2 if b == 'matharena' else k}",
                         "ent_first256": dd + rng.normal(), "ent_first1024": dd + 0.4 * rng.normal(),
                         "lp_first256": rng.normal(), "lp_first1024": -dd + 0.5 * rng.normal(), "ent_n_tokens": 1024.0,
                         "ent_closed": 0.0, "ent_degenerate": 0.0, "ent_prompt_tokens": 100.0 + k,
                         "ent_task_tokens": 90.0 + k, "ent_truncated": 0.0})
    work = tmp_path / "work"
    work.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(work / "entropy.parquet", index=False)
    out = str(tmp_path / "out.json")
    sem = semantics or S.entropy_semantics({"kinds": {"entropy": entropy_kind()}})
    H.save_json(out, {"ingest": {"features_digest": "main", "per_benchmark": {}}, "signs": {"main": True},
                      "verdict": {"call": "main"},
                      S.ENTROPY_KIND: {"rule": S.entropy_rule_record(),
                                       "rule_first": {"digest": S.entropy_rule_digest()},
                                       "ingest": {"features_digest": S.entropy_digest(str(work)), "semantics": sem,
                                                  "completeness": complete_export()}}})
    target = {b: {i: diff[(b, i)] for i in items[b]} for b in S.PARENTS}
    monkeypatch.setattr(S, "targets", lambda work, refresh=False: (target, None, {"t": 1}))
    monkeypatch.setattr(S.ICE, "load_items", lambda b: items[b])
    return SimpleNamespace(work=str(work), out=out, boots=60, refresh_targets=False, job="entropy"), items, diff


def test_the_entropy_signs_stage_reads_every_parent(tmp_path, monkeypatch, capsys):
    args, items, diff = _entropy_work(tmp_path, monkeypatch)
    S.stage_signs(args)
    st = H.load_json(args.out)
    assert st["signs"] == {"main": True} and "heads" not in st[S.ENTROPY_KIND]    # the rubric's signs untouched
    sg = st[S.ENTROPY_KIND]["signs"]
    f = sg["features"]
    assert set(f) == set(S.ENTROPY_COLUMNS) and sg["features_digest"] == S.entropy_digest(args.work)
    p = f["ent_first1024"]
    assert p["declared_sign"] == 1 and p["agreement_prong"]["positive"] == 4
    assert p["agreement_prong"]["declared_sign_ok"]
    assert p["agreement_prong"]["fields"]["matharena"] == "partial2_spearman"          # net of position there too
    assert p["units"]["multi_swebench"]["spearman_within"]["est"] > 0.5
    assert f["lp_first1024"]["declared_sign"] == -1 and f["lp_first1024"]["agreement_within"]["positive"] == 4
    assert f["ent_prompt_tokens"]["declared_sign"] == 0 and not f["ent_n_tokens"]["units"]["matharena"]["applies"]
    assert sg["registry"]["ent_first1024"]["primary"]
    with pytest.raises(SystemExit):                                  # the verdict needs the harness too
        S.stage_verdict(args)
    capsys.readouterr()
    H.save_json(str(tmp_path / "shown.json"), {S.ENTROPY_KIND: H.load_json(args.out)[S.ENTROPY_KIND]})
    S.stage_show(SimpleNamespace(out=str(tmp_path / "shown.json")))
    shown = capsys.readouterr().out
    assert "## the entropy job" in shown and "| ent_first1024 | 1 |" in shown and "| lp_first1024 | -1 |" in shown
    # statistics that are not the raw distribution's: every feature is read as a diagnostic, excluded
    failed = S.entropy_semantics({"kinds": {"entropy": entropy_kind(
        token_stats={"logprobs": "raw", "recorder_check": {"status": "FAILED"}})}})
    args, _, _ = _entropy_work(tmp_path / "f", monkeypatch, n=24, semantics=failed)
    S.stage_signs(args)
    f = H.load_json(args.out)[S.ENTROPY_KIND]["signs"]["features"]
    assert f["ent_first1024"]["declared_sign"] == 0 and "recorder" in f["ent_first1024"]["excluded"]


def test_the_entropy_stages_need_its_ingest(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    args = SimpleNamespace(work=str(work), out=str(tmp_path / "o.json"), job="entropy", boots=10,
                           refresh_targets=False)
    for stage in (S.stage_signs, S.stage_consistency, S.stage_verdict):
        with pytest.raises(SystemExit):
            stage(args)
    pd.DataFrame({"benchmark": ["matharena"], "item_id": ["a"], "ent_first1024": [1.0]}).to_parquet(
        work / "entropy.parquet", index=False)
    H.save_json(args.out, {"ingest": {}})
    with pytest.raises(SystemExit):                                  # a table, but no entropy section in OUT
        S.stage_signs(args)


def test_the_reference_covers_the_entropy_primary(tmp_path, monkeypatch):
    args, items, diff = _entropy_work(tmp_path, monkeypatch)
    je = S.load_entropy(args.work)
    je.loc[:5, "ent_first1024"] = np.nan                            # degenerate or empty units
    je.to_parquet(os.path.join(args.work, "entropy.parquet"), index=False)
    covered, what = S.reference_covered(args.work, H.load_json(args.out), "entropy")
    assert covered == set(je["item_id"][6:]) and len(covered) == 154 and "ent_first1024" in what


def gate_entry(**kw):
    """A harness entry whose transferred nested line has the given compact figures (the others off)."""
    lines = {n: {**compact(tl=0.0, on=0), "gate_pass": False} for n in S.NESTED}
    lines["transferred nested"] = {**compact(**kw), "gate_pass": None}
    lines["per-pair s=0.5 from B7 (forced)"] = {**compact(on=None), "gate_pass": None}
    return {"lines": lines, "r_within_pair_tl": 0.3}


def complete_export():
    """entropy.ingest.completeness of a complete export."""
    return {"complete": True, "per_parent": {b: {"units_expected": 3, "units_covered": 3, "complete": True}
                                             for b in S.PARENTS}}


def test_the_entropy_verdict_keeps_only_the_declared_primary(tmp_path):
    entry = gate_entry
    ok = {"positive": 4, "units": 4, "declared_sign_ok": True}
    two = {"positive": 2, "units": 4, "declared_sign_ok": False}
    sem = S.entropy_semantics({"kinds": {"entropy": entropy_kind()}})
    j = pd.DataFrame({c: [1.0] for c in ("benchmark", "item_id", *S.ENTROPY_COLUMNS)})
    est = {"rule": S.entropy_rule_record(), "rule_first": {"digest": S.entropy_rule_digest()},
           "ingest": {"semantics": sem, "completeness": complete_export()},
           "harness": {"_meta": {}, "ent_first1024": entry(), "lp_first1024": entry(), "ent_first256": entry(tl=-0.001),
                       "ent_n_tokens": entry()},
           "signs": {"registry": S.entropy_registry(j, sem),
                     "features": {"ent_first1024": {"agreement_prong": ok}, "lp_first1024": {"agreement_prong": ok},
                                  "ent_first256": {"agreement_prong": ok}, "ent_n_tokens": {"agreement_prong": two}}},
           "consistency": {"attempts": {"cot/ent_first1024": {"spearman": 0.6}}, "test_retest": {}},
           "reference": {"covered_by": "entropy job", "covered_keys_in_rows": 9,
                         "r": {"r=0.3": {"transferred nested": {"tl": -0.0025}}}}}
    v = S.entropy_verdict(est)
    assert v["keep"] == ["ent_first1024"] and v["call"] == "KEEP ent_first1024" and v["primary_eligible"]
    assert v["exploratory_pass"] == ["lp_first1024"] and not v["features"]["lp_first1024"]["keep"]
    assert v["rule"] == S.entropy_rule_record() and v["features"]["ent_first1024"]["within_pair_r"] == 0.3
    assert v["rule_digest"] == v["rule_digest_stored"] == v["rule_digest_at_first_ingest"] == S.entropy_rule_digest()
    assert v["rule_unchanged"] and v["complete"] and v["final"]
    assert v["consistency (not gating)"]["attempts"]["cot/ent_first1024"]["spearman"] == 0.6
    assert v["reference (not gating)"]["transferred_nested_tl"] == {"r=0.3": -0.0025}
    # the gate and the prong read exactly as for the rubric's primaries
    assert v["features"]["ent_first1024"]["nested"] == S.verdict(
        {"harness": {"x": est["harness"]["ent_first1024"]}})["features"]["x"]["nested"]
    est["signs"]["features"]["ent_first1024"]["agreement_prong"] = two
    assert S.entropy_verdict(est)["call"] == "NULL: the declared primary ent_first1024 fails the sign prong"
    est["signs"]["features"]["ent_first1024"]["agreement_prong"] = ok
    est["harness"]["ent_first1024"] = entry(tl=-0.0019)
    assert S.entropy_verdict(est)["call"] == "NULL: the declared primary ent_first1024 fails the gate"
    est["harness"]["ent_first1024"] = entry()
    # another version or job config, or statistics that are not the raw ones: never kept
    est["ingest"]["semantics"] = S.entropy_semantics({"kinds": {"entropy": entropy_kind(version="e2.0")}})
    v = S.entropy_verdict(est)
    assert not v["keep"] and v["call"].startswith("NULL: version 'e2.0'")
    est["ingest"]["semantics"] = S.entropy_semantics({"kinds": {"entropy": entropy_kind(cfg="ffffffffffffffff")}})
    v = S.entropy_verdict(est)
    assert not v["keep"] and v["call"].startswith("NULL: job config 'ffffffffffffffff'") and not v["primary_eligible"]
    est["ingest"]["semantics"] = S.entropy_semantics({"kinds": {"entropy": entropy_kind(
        token_stats={"logprobs": "raw", "recorder_check": {"status": "FAILED"}})}})
    assert S.entropy_verdict(est)["call"].startswith("NOT READ")
    est["ingest"]["semantics"] = sem
    h = est["harness"].pop("ent_first1024")
    assert S.entropy_verdict(est)["call"].startswith("NO DATA")
    est["harness"]["ent_first1024"] = h
    # a partial export (a commit that stopped with exit 75): the call is labelled, not final
    part = complete_export()
    part["complete"], part["per_parent"]["matharena"] = False, {"units_expected": 3, "units_covered": 1,
                                                                "complete": False}
    est["ingest"]["completeness"] = part
    v = S.entropy_verdict(est)
    assert v["call"] == "PRELIMINARY (partial export, units covered/expected {'matharena': '1/3'}): KEEP ent_first1024"
    assert not v["complete"] and not v["final"] and v["keep"] == ["ent_first1024"]
    del est["ingest"]["completeness"]
    assert S.entropy_verdict(est)["call"].startswith("PRELIMINARY (completeness unknown")
    est["ingest"]["completeness"] = complete_export()
    # a rule other than the one ingested first (or stored): marked, never final
    first = est["rule_first"]
    est["rule_first"] = {"digest": "0" * 64}
    v = S.entropy_verdict(est)
    assert v["call"].startswith("RULE CHANGED (at the first ingest 0000000000000000") and not v["rule_unchanged"]
    assert not v["final"]
    est["rule_first"] = first
    est["rule"] = {**S.entropy_rule_record(), "primary": "ent_first256"}          # edited by hand: its digest fails
    assert S.entropy_verdict(est)["call"].startswith("RULE CHANGED")
    est["rule"] = S.entropy_rule_record()
    assert S.entropy_verdict(est)["final"]


def test_the_entropy_verdict_reads_only_what_ingest_recorded(tmp_path, monkeypatch):
    import copy
    args, items, diff = _entropy_work(tmp_path, monkeypatch)
    S.stage_signs(args)
    st = H.load_json(args.out)
    est = st[S.ENTROPY_KIND]
    joined = S.load_entropy(args.work)
    maps = S.entropy_harness_maps(joined, S.entropy_stage_registry(joined, est))
    est["harness"] = {"_meta": {}, **{n: {**gate_entry(), "x_digest": S.maps_digest({n: m})} for n, m in maps.items()}}
    H.save_json(args.out, st)
    va = SimpleNamespace(work=args.work, out=args.out, job="entropy")
    S.stage_verdict(va)
    st = H.load_json(args.out)
    v = st[S.ENTROPY_KIND]["verdict"]
    assert st["verdict"] == {"call": "main"} and v["call"] == "KEEP ent_first1024" and v["final"]
    assert v["provenance"] == {"ok": True, "problems": [], "features_digest": S.entropy_digest(args.work),
                               "harness_lines_checked": len(maps)}
    assert v["rule"]["primary"] == "ent_first1024" and v["rule_digest"] == S.entropy_rule_digest()
    assert v["script_digest"] == H.digest(["experiments/strong_llm_eval.py"])
    hist = st[S.ENTROPY_KIND]["history"]
    assert [h["stage"] for h in hist] == ["verdict"] and hist[0]["call"] == v["call"] and hist[0]["final"]
    assert hist[0]["rule_digest"] == v["rule_digest"] and hist[0]["primary_line"]["gate_pass_nested"]
    S.stage_verdict(va)                                              # every call appends
    good = H.load_json(args.out)
    assert len(good[S.ENTROPY_KIND]["history"]) == 2

    def refused(mutate, match, stage=S.stage_verdict, a=va):
        bad = copy.deepcopy(good)
        mutate(bad[S.ENTROPY_KIND])
        H.save_json(args.out, bad)
        with pytest.raises(SystemExit, match=match):
            stage(a)
        H.save_json(args.out, good)
    # signs of another table, a harness line on another x: the verdict refuses them
    refused(lambda e: e["signs"].update(features_digest="0123456789abcdef"), "entropy.signs were computed on 0123")
    refused(lambda e: e["harness"]["ent_first1024"].update(x_digest="feedfeedfeedfeed"), "computed on another x")
    refused(lambda e: e["harness"].update(ent_prompt_tokens=gate_entry()), "not a usable feature")
    # another --work beside this --out: every entropy stage refuses it
    other = tmp_path / "other_work"
    other.mkdir()
    S.load_entropy(args.work).assign(ent_first1024=lambda d: d.ent_first1024 + 1.0).to_parquet(
        other / "entropy.parquet", index=False)
    wa = SimpleNamespace(**{**vars(args), "work": str(other)})
    for stage in (S.stage_verdict, S.stage_signs, S.stage_harness, S.stage_consistency, S.stage_reference):
        with pytest.raises(SystemExit, match="ingest"):
            stage(SimpleNamespace(**vars(wa), feats=None))
    assert H.load_json(args.out) == good
    # a rule other than the first ingest's: refused, or read marked RULE CHANGED (never final)
    monkeypatch.setitem(S.ENTROPY_RULE, "primary", "ent_first256")
    with pytest.raises(SystemExit, match="not the rule of the first ingest"):
        S.stage_verdict(va)
    S.stage_verdict(SimpleNamespace(**vars(va), accept_rule_change=True))
    v = H.load_json(args.out)[S.ENTROPY_KIND]["verdict"]
    assert v["call"].startswith("RULE CHANGED") and not v["final"] and not v["rule_unchanged"]


def test_consistency_against_the_attempts_and_test_retest(tmp_path, monkeypatch):
    args, items, diff = _entropy_work(tmp_path, monkeypatch)
    je = S.load_entropy(args.work)
    mh = je[je.benchmark == "matharena"].reset_index(drop=True)
    rng = np.random.default_rng(5)
    x = mh["ent_first1024"].to_numpy()[:30]
    jm = pd.DataFrame({"benchmark": "matharena", "item_id": mh["item_id"][:30],
                       "att_cot_ent_first1024": x + 0.1 * rng.normal(size=30),
                       "att_cot_tok_entropy": rng.normal(size=30), "att_cot_k": 4.0, "rubric_a": 1.0})
    jm = pd.concat([jm, pd.DataFrame({"benchmark": ["multi_swebench"], "item_id": ["mul0000"],
                                      "att_cot_ent_first1024": [9.0]})], ignore_index=True)
    c = S.consistency(je, jm, {"logprobs": "raw"}, {"logprobs": "raw"})
    a = c["attempts"]["cot/ent_first1024"]
    want = pd.Series(x).corr(jm["att_cot_ent_first1024"][:30].reset_index(drop=True), method="spearman")
    assert a["n_items"] == a["n_units"] == 30 and a["spearman"] == pytest.approx(want, abs=1e-4) and want > 0.8
    assert abs(c["attempts"]["cot/tok_entropy"]["spearman"]) < 0.5
    assert c["same_definition"] and not c["gating"]
    # per prompt (the attempts' unit is the text): the mean over a prompt's units against the mean attempt value
    xp = x.reshape(15, 2).mean(1)
    yp = jm["att_cot_ent_first1024"][:30].to_numpy().reshape(15, 2).mean(1)
    assert a["n_prompts"] == 15 and a["spearman_prompts"] == pytest.approx(
        pd.Series(xp).corr(pd.Series(yp), method="spearman"), abs=1e-4)
    # test-retest: matharena's prompts come in 20 pairs of identical text under other keys
    tr = c["test_retest"]
    assert set(tr) == {"matharena"} and tr["matharena"]["pairs"] == 20
    e = mh["ent_first1024"].to_numpy()
    assert tr["matharena"]["spearman"] == pytest.approx(
        pd.Series(e[0::2]).corr(pd.Series(e[1::2]), method="spearman"), abs=1e-4)
    # ICC(1) over every unit of the repeated prompts: for pairs, (MSB - MSW) / (MSB + MSW)
    P = np.stack([e[0::2], e[1::2]], 1)
    msb = 2 * ((P.mean(1) - P.mean()) ** 2).sum() / 19
    msw = ((P - P.mean(1, keepdims=True)) ** 2).sum() / 20
    assert tr["matharena"]["icc1"] == pytest.approx((msb - msw) / (msb + msw), abs=1e-4)
    assert tr["matharena"]["units_in_repeated_prompts"] == 40
    # unequal groups: every unit counts (a triple is not cut to its first two)
    assert S.icc1([[1.0, 1.1, 0.9], [2.0, 2.1], [3.0, 2.9], [4.0, 4.2], [5.0, 5.1]]) > 0.95
    assert S.icc1([[1.0, 2.0]] * 4) is None                                  # under CONSISTENCY_MIN groups
    # one unit copied to two item_ids (one key) is not a pair; other semantics are not the same definition
    je.loc[je.item_id == "mat0001", "key"] = "kmat0000"
    c = S.consistency(je, jm, {"logprobs": "processed"}, {"logprobs": "raw"})
    assert c["test_retest"]["matharena"]["pairs"] == 19 and not c["same_definition"]
    assert c["attempts"]["cot/ent_first1024"]["n_units"] == 29
    assert S.consistency(je)["attempts"] is None
    # the stage stores it under entropy with both digests
    jm.to_parquet(os.path.join(args.work, "features.parquet"), index=False)
    S.stage_consistency(args)
    st = H.load_json(args.out)
    cs = st[S.ENTROPY_KIND]["consistency"]
    assert cs["main_features_digest"] == S.features_digest(args.work) and cs["features_digest"] == S.entropy_digest(
        args.work)
    assert cs["attempts"]["cot/ent_first1024"]["n_items"] == 30 and st["signs"] == {"main": True}


def test_run_facts_for_the_entropy_session(tmp_path):
    kdir = tmp_path / "kaggle_d"
    (kdir / "_detail").mkdir(parents=True)
    t0 = 1_790_100_000.0
    man = {"slug": "M", "wall_s": 20000.0, "notebook_digest": "e" * 64, "sessions": 2,
           "kinds": {"entropy": entropy_kind()}}
    json.dump(man, open(kdir / "manifest.json", "w"))
    # 12 units in 3 shards of 4, written 600 s apart
    pd.DataFrame({"unit": [f"u{k}" for k in range(12)], "benchmark": [S.PARENTS[k % 4] for k in range(12)],
                  "item_ids": [[f"i{k}"] * (1 + (k == 0)) for k in range(12)],
                  "t": [t0 + 1000 + 600 * (k // 4) for k in range(12)], "n_tokens": 1024, "prompt_tokens": 500,
                  "capped": True, "closed": False, "degenerate": False, "truncated": [k == 3 for k in range(12)]}
                 ).to_parquet(kdir / "_detail" / "entropy_units.parquet", index=False)
    root = {"script_sha256": "e" * 64, "wall_s_total": 20000.0,
            "sessions": [{"run": "first", "t0": t0 - 40000, "wall_s": 17000.0, "stats": {"rubric": {"secs": 6000.0}}},
                         {"run": "d", "t0": t0, "wall_s": 3000.0, "status": {"entropy": "deadline"},
                          "stats": {"entropy": {"secs": 1800.0, "gen_tokens": 12 * 1024, "units_done_session": 12,
                                                "shards": 3, "gen_tok_s": 6.8, "recorder_check": {"status": "ok"}}}}],
            "plan": {"M": {"entropy": {"units": 20, "hours_est": 1.0, "rates_tok_s": {"decode": 180,
                                                                                      "decode_slow": 135},
                                       "units_per_commit_slow": 20}}},
            "progress": {"M": {"entropy": {"units": 20, "done": 12}}}}
    (tmp_path / "raw" / "d" / "strong_probe").mkdir(parents=True)
    path = str(tmp_path / "raw" / "d" / "strong_probe" / "manifest.json")
    json.dump(root, open(path, "w"))
    f = S.run_facts(str(kdir), path)
    tl = f["timeline"]
    assert tl["entropy_units"] == 12 and tl["entropy_item_ids"] == 13 and tl["entropy_shards"] == 3
    assert tl["entropy_shard_s"]["median"] == 600.0 and tl["entropy_units_per_shard"] == 4.0
    assert tl["entropy_gen_tokens"] == 12 * 1024 and tl["entropy_prompt_tokens"] == 6000
    assert tl["entropy_truncated_share"] == round(1 / 12, 4) and tl["entropy_capped_share"] == 1.0
    assert tl["entropy_units_per_benchmark"] == {b: 3 for b in S.PARENTS}
    assert f["sessions"][1]["entropy"]["units_done_session"] == 12 and "entropy" not in f["sessions"][0]
    der = f["derived"]
    assert der["entropy_gen_tok_s"] == round(12 * 1024 / 1800, 1)
    assert der["entropy_gen_tok_s_over_plan_decode"] == round(12 * 1024 / 1800 / 180, 3)
    assert der["entropy_hours_per_unit_over_plan"] == round(1800 / 12 / (3600 / 20), 3)
    assert der["entropy_units_left"] == 8 and der["entropy_hours_left_at_measured"] == round(8 * 1800 / 12 / 3600, 2)
    assert f["plan"]["entropy"]["units_per_commit_slow"] == 20
    # the stage writes entropy.run and leaves the first commit's run as it was
    out = tmp_path / "out.json"
    H.save_json(str(out), {"run": {"first": True}})
    S.stage_run(SimpleNamespace(kaggle=[str(kdir)], run_manifest=path, out=str(out)))
    st = H.load_json(str(out))
    assert st["run"] == {"first": True} and st[S.ENTROPY_KIND]["run"]["timeline"]["entropy_units"] == 12


# --- the kit's entropy job, end to end ---------------------------------------------------------------------

def _pairs_ready(data, items):
    """The kit's synthetic data with what paiec.data.load_pairs reads besides: benchmark_id in response.parquet,
    subject_id in subjects.parquet."""
    from paiec import data as D
    for b in items:
        p = os.path.join(data, b)
        r = pd.read_parquet(os.path.join(p, "response.parquet"))
        r["benchmark_id"] = b
        r.to_parquet(os.path.join(p, "response.parquet"), index=False)
        pd.DataFrame({"subject_id": sorted(set(r.subject_id)), **{f: "x" for f in D.SUBJECT_FIELDS}}).to_parquet(
            os.path.join(p, "subjects.parquet"), index=False)


def test_the_kits_entropy_job_end_to_end(tmp_path, monkeypatch):
    """kaggle/strong_probe/strong_probe.py on its mock backend (tests/test_kaggle_probe.py) and synthetic items:
    the first commit's export ingested, then commit D (the entropy job with the first commit's Output attached)
    exported, schema-checked and ingested; then an entropy-only commit, alone and beside the first export."""
    from paiec import data as D
    from tests import test_kaggle_probe as KP
    SP = KP.SP
    data = tmp_path / "data"
    items = KP.write_synthetic(str(data))
    _pairs_ready(str(data), items)
    monkeypatch.setattr(D, "DATA_DIR", str(data))
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    first = tmp_path / "first"
    commit = tmp_path / "commit_d"
    # the kit's tests run with --task-tokens 256, another job config than the one the rule was fixed for (the
    # kit's defaults): the rule is pinned to this one here, as it is to the defaults for commit D
    cfg = SP.digest(SP.job_cfg(SP.parse_args(KP.entropy_args(data, commit)), "entropy"))
    assert cfg != S.ENTROPY_CFG
    monkeypatch.setitem(S.ENTROPY_RULE["config"], "cfg", cfg)
    assert SP.main(KP.run_args(data, first), backend=KP.MockBackend(), now=KP.clock()) == 0
    ex1 = first / slug / "export"
    args = _ing(tmp_path, ex1)
    S.stage_ingest(args)
    st = H.load_json(args.out)
    st.update(signs={"kept": 1}, attempts={"decision": "GO"}, verdict={"call": "NULL"})
    H.save_json(args.out, st)
    before = H.load_json(args.out)
    be = KP.MockBackend()
    assert SP.main(KP.resumed(KP.entropy_args(data, commit), first), backend=be, now=KP.clock()) == 0
    assert be.entropy_prompts and not be.rubric_prompts
    ex = commit / slug / "export"
    rep = S.check_schema(str(ex), benches=SP.BENCHES)
    assert rep["ok"] and not rep["warnings"], (rep["errors"], rep["warnings"])
    S.stage_check_schema(_ing(tmp_path, ex))                         # exits 1 on any error
    S.stage_ingest(_ing(tmp_path, ex))
    st = H.load_json(args.out)
    assert _main_part(st) == before                                  # the rubric's and the attempts' results kept
    e = st[S.ENTROPY_KIND]["ingest"]
    assert e["main_features"]["status"].startswith("byte-identical")
    for b, rows in items.items():
        r = e["per_benchmark"][b]
        assert r["rows"] == r["hash_ok"] == r["covered_items"] == len(rows) and r["hash_mismatch"] == 0, (b, r)
    sem = e["semantics"]
    assert sem["readable"] and sem["primary_eligible"] and sem["version"] == SP.ENTROPY_VERSION == S.ENTROPY_VERSION
    assert sem["cfg"] == cfg and e["completeness"]["complete"]
    assert {b: v["units_covered"] for b, v in e["completeness"]["per_parent"].items()} == {
        "matharena": 7, "multi_swebench": 2, "real_webagents": 1, "researchcodebench": 1}
    assert sem["recorder_check"] == "ok" and not sem["sign_disagreements"]
    assert sem["primary_manifest"] == S.ENTROPY_PRIMARY
    je = S.load_entropy(args.work)
    assert len(je) == sum(map(len, items.values())) and np.isfinite(je[S.ENTROPY_PRIMARY]).all()
    en = pd.read_parquet(ex / "entropy" / "entropy.parquet").set_index("item_id")
    assert np.allclose(je.set_index("item_id").loc[en.index, S.ENTROPY_PRIMARY], en[S.ENTROPY_PRIMARY])
    reg = S.entropy_registry(je, sem)
    assert {c for c, v in reg.items() if v["usable"]} == set(S.ENTROPY_SIGNS) == set(SP.ENTROPY_EXPORT) | {
        "ent_n_tokens", "ent_closed"}
    assert all(reg[f]["sign"] == sg for f, (sg, _) in SP.ENTROPY_EXPORT.items())
    assert SP.ENTROPY_PRIMARY == S.ENTROPY_PRIMARY
    # the consistency check reads both: matharena's four attempted items, too few for a correlation; the two probe
    # items share a text under other competitions, so they are two units with one prompt: one test-retest pair
    S.stage_consistency(SimpleNamespace(work=args.work, out=args.out))
    c = H.load_json(args.out)[S.ENTROPY_KIND]["consistency"]
    assert c["same_definition"] and c["attempts"]["cot/ent_first1024"]["n_items"] == 4
    assert c["attempts"]["cot/ent_first1024"]["spearman"] is None and c["test_retest"]["matharena"]["pairs"] == 1
    # the session's facts from the kit's own root manifest go to entropy.run; `run` is not written
    S.stage_run(SimpleNamespace(kaggle=[str(ex)], run_manifest=str(commit / "manifest.json"), out=args.out))
    st = H.load_json(args.out)
    n_units = len(SP.entropy_units(str(data), SP.PARENTS))
    er = st[S.ENTROPY_KIND]["run"]
    assert "run" not in st and er["timeline"]["entropy_units"] == n_units and er["timeline"]["entropy_shards"] >= 1
    assert er["sessions"][-1]["entropy"]["units_done_session"] == n_units and "entropy" not in er["sessions"][0]
    assert er["sessions"][-1]["entropy"]["recorder_check"]["status"] == "ok"
    assert er["derived"]["entropy_units_left"] == 0 and er["timeline"]["entropy_gen_tokens"] > 0
    H.save_json(str(tmp_path / "shown.json"), {S.ENTROPY_KIND: st[S.ENTROPY_KIND]})
    S.stage_show(SimpleNamespace(out=str(tmp_path / "shown.json")))
    # an entropy-only commit (nothing attached): alone, and beside the first commit's export
    only = tmp_path / "only"
    assert SP.main(KP.entropy_args(data, only), backend=KP.MockBackend(), now=KP.clock()) == 0
    ex3 = only / slug / "export"
    assert S.check_schema(str(ex3), benches=SP.BENCHES)["ok"]
    S.stage_ingest(_ing(tmp_path, ex3))
    assert _main_part(H.load_json(args.out)) == before
    S.stage_ingest(_ing(tmp_path, ex1, ex3))
    st = H.load_json(args.out)
    assert _main_part(st) == before and st[S.ENTROPY_KIND]["meta"]["kaggle_dir"] == str(ex3)
    assert "consistency" in st[S.ENTROPY_KIND]                      # the same entropy table: kept
