"""experiments/strong_llm_eval.py on synthetic Kaggle-like shards (no data, no
model): the schema check, the join and its hash verification, the attempt
aggregation, the declared signs, the leave-one-parent-out head, the attempt
call (on every attempted text and on the probe texts only), the gate verdict
and the run's facts."""
import hashlib
import json
import math
import os

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
