"""paiec/pool.py on synthetic pools (tiny formatted_tables in tmp_path, synthetic
item text only), plus RP4 on the legacy five when their data is present.

What is pinned: the manifest check; the first 0/1 response per item in file
order with nulls and other values dropped; the caps and their orders; the
selection rule; the families and whole-family exclusion; Mode A items read by
column projection in bounded batches; the evaluation loader against a direct
read; the bundle arithmetic against paiec.prior (the synthetic RP4, and RP4
proper on the legacy five); C1m's matching; the EB iteration; byte-stable
bundles; that the evaluation loader is the only code a post-lock fix may touch.
"""
import ast
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from paiec import pool as P
from paiec import prior as PR
from paiec.evaluator import stable_hash
from paiec.hier import HierPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = "SYNTHETIC-ITEM-TEXT"
DISPLAY = "SECRET-DISPLAY-NAME"
NAMES = ["acme large 70b", "acme small 7b", "zeta pro", "omni one", "nova two", "quill", "rune"]


# --- a synthetic pool --------------------------------------------------------------------

def _scale(direction="higher_is_better"):
    return json.dumps({"direction": direction, "kind": "discrete", "values": [0.0, 1.0]})


def bench(name, n_subjects=6, n_items=100, seed=0, kind="binary", direction="higher_is_better",
          release="public", source_url=None, paper_url=None, names=None, interactors=None,
          shared_items=True, repeats=1, description="", row_group=None, extra_rows=()):
    """A benchmark spec: catalogue row, subjects, responses (file order), items."""
    rng = np.random.default_rng(seed)
    names = names or NAMES
    subjects = []
    for s in range(n_subjects):
        nm = names[s % len(names)]
        subjects.append((f"{name}-s{s}", {"normalized_name": nm, "provider": "acme",
                                           "release_date": f"2024-0{1 + s % 9}-15" if s % 3 else "",
                                           "access_date": "", "harness": "h", "harness_version": "",
                                           "reasoning_effort": "", "subject_features_extra": ""}))
    items = [(f"{name}-i{i}", f"{TEXT} {name} {i}\nA) one\nB) two\nC) three\nD) four",
              f"tier={'ab'[i % 2]}") for i in range(n_items * (1 if shared_items else n_subjects))]
    rows = []
    for s, (sid, _) in enumerate(subjects):
        th = rng.normal(0, 1)
        mine = items if shared_items else items[s * n_items:(s + 1) * n_items]
        for _ in range(repeats):
            for j, (iid, _, _) in enumerate(mine):
                y = float(rng.random() < 1 / (1 + np.exp(-(th - 0.02 * (j % 50)))))
                inter = None if interactors is None else interactors(sid, iid)
                rows.append((sid, iid, y, inter))
    rows += list(extra_rows)
    return {"name": name, "kind": kind, "direction": direction, "release": release,
            "source_url": source_url or f"https://example.org/{name}/repo",
            "paper_url": paper_url or f"https://arxiv.org/abs/2401.{stable_hash('paper', name) % 90000 + 10000:05d}",
            "description": description, "subjects": subjects, "items": items, "rows": rows,
            "row_group": row_group}


def write_pool(root, specs, revision=P.REVISION):
    """Write formatted_tables and a manifest for `specs`; returns root."""
    man = {"repo": "synthetic", "revision": revision, "fetched_utc": "2026-10-08T00:00:00Z",
           "tables": ["benchmarks", "subjects", "responses", "items"], "items_skipped": [],
           "benchmarks": {}}
    for sp in specs:
        b = sp["name"]
        d = os.path.join(root, b, "formatted_tables")
        os.makedirs(d, exist_ok=True)
        cat = {"benchmark_id": [b], "name": [b], "release": [sp["release"]], "granularity": ["item"],
               "response_type": [sp["kind"]], "response_scale": [_scale(sp["direction"])],
               "source_url": [sp["source_url"]], "paper_url": [sp["paper_url"]],
               "one_line_description": [""], "description": [sp["description"]],
               "benchmark_features": [""]}
        pq.write_table(pa.table(cat), os.path.join(d, "benchmarks.parquet"))
        files = ["benchmarks.parquet"]
        if sp["release"] == "public":
            subj = {"subject_id": [s for s, _ in sp["subjects"]],
                    "display_name": [DISPLAY] * len(sp["subjects"])}
            for f in P.SUBJECT_FIELDS:
                subj[f] = [v[f] or None for _, v in sp["subjects"]]
            pq.write_table(pa.table(subj), os.path.join(d, "subjects.parquet"))
            rows = sp["rows"]
            resp = {"subject_id": [r[0] for r in rows], "item_id": [r[1] for r in rows],
                    "response_id": [f"r{k}" for k in range(len(rows))],
                    "benchmark_id": [b] * len(rows), "trial": [1] * len(rows),
                    "test_condition": [None] * len(rows),
                    "interactors": pa.array([r[3] for r in rows], pa.string()),
                    "response": pa.array([r[2] for r in rows], pa.float64())}
            pq.write_table(pa.table(resp), os.path.join(d, "responses.parquet"),
                           row_group_size=sp["row_group"])
            it = sp["items"]
            items = {"item_id": [i for i, _, _ in it], "benchmark_id": [b] * len(it),
                     "raw_item_id": [i for i, _, _ in it], "content": [c for _, c, _ in it],
                     "asset_manifest": [None] * len(it),
                     "grading_criterion": [json.dumps({"reference_answer": TEXT, "rule": "x",
                                                       "response_scale": {"kind": "discrete",
                                                                          "values": [0, 1]}})] * len(it),
                     "verifier": [TEXT * 3] * len(it), "content_hash": ["h"] * len(it),
                     "item_features": [f for _, _, f in it]}
            pq.write_table(pa.table(items), os.path.join(d, "items.parquet"), row_group_size=37)
            files += ["subjects.parquet", "responses.parquet", "items.parquet"]
        rec = {"files": {}, "release": sp["release"], "granularity": "item",
               "response_type": sp["kind"], "response_scale": _scale(sp["direction"]),
               "status": "ok" if sp["release"] == "public" else "not public"}
        for f in files:
            p = os.path.join(d, f)
            rec["files"][f] = {"bytes": os.path.getsize(p),
                               "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
        man["benchmarks"][b] = rec
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    return str(root)


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    """Caches in tmp_path; in-process caches emptied."""
    monkeypatch.setattr(P, "FITS_DIR", str(tmp_path / "fits"))
    monkeypatch.setattr(P, "EVAL_DIR", str(tmp_path / "eval"))
    for c in (P._RESP, P._VERIFIED, P._MANIFESTS):
        c.clear()
    yield
    for c in (P._RESP, P._VERIFIED, P._MANIFESTS):
        c.clear()


def standard_specs():
    return [bench("alpha", seed=1, source_url="https://huggingface.co/datasets/Fam/alpha"),
            bench("beta", seed=2, source_url="https://huggingface.co/datasets/Fam/beta"),
            bench("gamma", seed=3, paper_url="https://arxiv.org/pdf/2402.04249v2"),
            bench("delta", seed=4, paper_url="https://arxiv.org/abs/2402.04249"),
            bench("eps", n_subjects=1, seed=5),
            bench("lowbet", seed=6, direction="lower_is_better"),
            bench("mixok", seed=7, kind="mixed",
                  extra_rows=[("mixok-s0", "mixok-i1", 0.5, None)]),
            bench("contin", seed=8, kind="continuous"),
            bench("hidden", seed=9, release="withheld")]


@pytest.fixture
def pool(tmp_path):
    return write_pool(tmp_path / "pool", standard_specs())


# --- verification --------------------------------------------------------------------------

def test_verify_raises_on_a_changed_byte_and_on_a_missing_entry(tmp_path):
    root = write_pool(tmp_path / "pool", [bench("alpha")])
    path = P.verify("alpha", "responses", root)
    data = bytearray(open(path, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(path, "wb").write(bytes(data))
    with pytest.raises(P.PoolError, match="sha256"):
        P.verify("alpha", "responses", root)
    with pytest.raises(P.PoolError, match="not in the manifest"):
        P.verify("alpha", "nothing.parquet", root)
    with pytest.raises(P.PoolError, match="not in the manifest"):
        P.verify("absent", "responses", root)
    with open(path, "ab") as f:
        f.write(b"x")
    with pytest.raises(P.PoolError, match="bytes"):
        P.verify("alpha", "responses", root)


def test_the_revision_is_pinned(tmp_path):
    root = write_pool(tmp_path / "pool", [bench("alpha")], revision="0" * 40)
    with pytest.raises(P.PoolError, match="revision"):
        P.manifest(root)


# --- responses and pair counts ----------------------------------------------------------------

def test_first_01_response_in_file_order_nulls_and_other_values_dropped(tmp_path):
    rows = []
    s0, s1 = "one-s0", "one-s1"
    # s0: 90 items; item 1 again later in the file (1 after its first 0), then 0.5 and null
    for i in range(90):
        rows.append((s0, f"one-i{i}", float(i % 3 == 0), None))
    rows += [(s0, "one-i1", 0.5, None), (s0, "one-i1", None, None)]
    rows.insert(5, (s0, "one-i1", 1.0, None))
    # s1: 79 distinct 0/1 items, nulls and 0.5 on 41 others: not eligible
    for i in range(79):
        rows.append((s1, f"one-i{i}", 1.0, None))
    rows += [(s1, f"one-i{i}", None, None) for i in range(79, 120)]
    rows += [(s1, f"one-i{i}", 0.5, None) for i in range(79, 120)]
    sp = bench("one", n_subjects=2, n_items=120)
    sp["rows"] = rows
    root = write_pool(tmp_path / "pool", [sp])
    t = P.pair_table("one", root)
    assert t["rows"] == len(rows)
    assert t["nonnull"] == len(rows) - 1 - 41
    pairs = P.pair_counts("one", root)
    assert [p[0] for p in pairs] == [s0]                    # s1 has 79 items
    sid, n, k, lg = pairs[0]
    first = {}
    for s, i, y, _ in rows:
        if s == s0 and y in (0.0, 1.0) and i not in first:
            first[i] = y
    assert n == len(first) == 90
    assert k == int(sum(first.values()))
    assert lg == pytest.approx(np.log((k + 0.5) / (n - k + 0.5)), abs=1e-15)
    assert t["level_cc"] == pytest.approx(lg)
    assert P.levels(["one"], root) == {"one": t["level_cc"]}
    fit = P.load_fit_pairs("one", root=root)
    assert len(fit) == 1 and len(fit[0].responses) == 90
    assert {r.item_key: r.label for r in fit[0].responses} == {i: int(y) for i, y in first.items()}
    assert [r.item_key for r in fit[0].responses] == list(first)        # file order


def test_pair_tables_cache_holds_no_item_text(pool):
    P.pair_tables(["alpha", "beta"], 1, pool)
    P.load_fit_pairs("alpha", root=pool, content=True)
    P.eval_index("alpha", pool)
    for d in (P.FITS_DIR, P.EVAL_DIR):
        for dp, _, fn in os.walk(d):
            for f in fn:
                raw = open(os.path.join(dp, f), "rb").read()
                if f.endswith(".npz"):
                    import zipfile
                    with zipfile.ZipFile(os.path.join(dp, f)) as z:
                        raw = b"".join(z.read(n) for n in z.namelist())
                assert TEXT.encode() not in raw and DISPLAY.encode() not in raw, f


# --- caps -----------------------------------------------------------------------------------------

def test_per_pair_item_cap_keeps_a_sparse_benchmarks_pairs(tmp_path):
    sparse = bench("sparse", n_subjects=6, n_items=100, shared_items=False, seed=1)
    root = write_pool(tmp_path / "pool", [sparse])
    pairs = P.load_fit_pairs("sparse", k=100, m_p=90, root=root)
    assert len(pairs) == 6 and all(len(p.responses) == 90 for p in pairs)
    # a benchmark-wide cap of 90 would leave each pair < 80 items
    rank = P.item_rank("sparse", [i for i, _, _ in sparse["items"]])
    top = set(sorted(rank, key=rank.get)[:90])
    assert all(sum(r.item_key in top for r in p.responses) < 80 for p in pairs)
    for p in pairs:
        mine = sorted({r.item_key for r in p.responses} | {i for s, i, _, _ in sparse["rows"]
                                                           if s == p.subject_id}, key=rank.get)
        assert {r.item_key for r in p.responses} == set(mine[:90])


def test_per_pair_item_cap_equals_a_benchmark_wide_cap_on_a_dense_benchmark(tmp_path):
    dense = bench("dense", n_subjects=5, n_items=150, seed=2)
    root = write_pool(tmp_path / "pool", [dense])
    pairs = P.load_fit_pairs("dense", k=100, m_p=100, root=root)
    rank = P.item_rank("dense", [i for i, _, _ in dense["items"]])
    top = set(sorted(rank, key=rank.get)[:100])
    assert all({r.item_key for r in p.responses} == top for p in pairs)


def test_round_robin_pair_order():
    subs = [(f"x{j}", {"normalized_name": nm}) for j, nm in
            enumerate(["a one", "a one", "a one", "b two", "c three", "", ""])]
    o1 = P.cap_order("bench1", subs)
    assert o1 == P.cap_order("bench1", list(reversed(subs)))       # deterministic, order-free
    names = {s: (P.PR.canon_name(d) or s) for s, d in subs}
    # a name with several pairs gets one slot per round
    firsts = o1[:len(set(names.values()))]
    assert len({names[s] for s in firsts}) == len(firsts)
    assert [names[s] for s in o1].count("a one") == 3
    assert o1.index([s for s in o1 if names[s] == "a one"][1]) >= len(set(names.values()))
    # the same names lead on two benchmarks
    subs2 = [(f"y{j}", d) for j, (_, d) in enumerate(subs[:5])]
    lead1 = [names[s] for s in o1 if names[s] in ("a one", "b two", "c three")][:3]
    o2 = P.cap_order("bench2", subs2)
    names2 = {s: P.PR.canon_name(d) for s, d in subs2}
    assert [names2[s] for s in o2][:3] == lead1
    assert lead1 == sorted(["a one", "b two", "c three"], key=lambda c: stable_hash("pool-name", c))


def test_pair_cap_keeps_k_pairs_in_cap_order(tmp_path):
    sp = bench("many", n_subjects=9, n_items=90, seed=3)
    root = write_pool(tmp_path / "pool", [sp])
    pairs = P.load_fit_pairs("many", k=4, root=root)
    order = P.cap_order("many", [(s, {**d, "_sid": s}) for s, d in sp["subjects"]])
    assert {p.subject_id for p in pairs} == set(order[:4])
    file_order = [s for s, _ in sp["subjects"]]
    assert [p.subject_id for p in pairs] == [s for s in file_order if s in set(order[:4])]


# --- selection -------------------------------------------------------------------------------------

def test_selection_rule(pool):
    L, U, rep = P.select(pool)
    assert L == ["alpha", "beta", "delta", "eps", "gamma", "lowbet", "mixok"]
    assert "contin" not in L and "hidden" not in L                    # type, release
    assert rep["flagged_by_direction"] == ["lowbet"]
    Lf, Uf, _ = P.select(pool, flagged=("lowbet",))
    assert "lowbet" in Lf and "lowbet" not in Uf
    assert rep["benchmarks"]["mixok"]["share01"] == pytest.approx(600 / 601)


def test_mixed_below_the_share_is_dropped(tmp_path):
    bad = [("mixlow-s0", f"mixlow-i{i}", 0.5, None) for i in range(40)]
    root = write_pool(tmp_path / "pool", [bench("mixlow", kind="mixed", extra_rows=bad),
                                          bench("mixhi", kind="mixed")])
    L, _, rep = P.select(root)
    assert L == ["mixhi"]
    assert rep["benchmarks"]["mixlow"]["share01"] == pytest.approx(600 / 640)


def test_a_benchmark_without_an_eligible_pair_is_dropped(tmp_path):
    root = write_pool(tmp_path / "pool", [bench("tiny", n_items=50), bench("ok")])
    assert P.select(root)[0] == ["ok"]


def test_check_csvs_agrees_and_catches_a_difference(pool, tmp_path):
    import csv
    L, _, rep = P.select(pool)
    names = sorted(b for b, v in rep["benchmarks"].items() if "eligible_pairs" in v)
    bpath, ppath = tmp_path / "benchmarks.csv", tmp_path / "pairs.csv"
    with open(bpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "release", "granularity", "response_type", "share01",
                    "eligible_pairs", "level_cc"])
        for b in names:
            t = P.pair_table(b, pool)
            w.writerow([b, "public", "item", rep["benchmarks"][b]["kind"], repr(t["share01"]),
                        len(t["pairs"]), repr(t["level_cc"]) if t["pairs"] else ""])
    with open(ppath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "subject_id", "n_items", "k_first", "logit_cc"])
        for b in names:
            for s, n, k, lg in P.pair_counts(b, pool):
                w.writerow([b, s, n, k, repr(lg)])
    got = P.check_csvs(names, pool, str(bpath), str(ppath))
    assert got["max_abs_diff"]["level"] == 0
    assert got["tol"] == P.CSV_TOL == 1e-12
    assert P.check_selection_csv(L, str(bpath))["equal"]
    # a level 1e-11 off is a difference at the default tolerance (it was not at 1e-9)
    orig = open(bpath).read()
    lines = orig.splitlines()
    for j, line in enumerate(lines[1:], 1):
        parts = line.split(",")
        if parts[-1]:
            parts[-1] = repr(float(parts[-1]) + 1e-11)
            lines[j] = ",".join(parts)
            break
    open(bpath, "w").write("\n".join(lines) + "\n")
    with pytest.raises(P.PoolError, match="above 1e-12"):
        P.check_csvs(names, pool, str(bpath), str(ppath))
    assert P.check_csvs(names, pool, str(bpath), str(ppath), tol=1e-9)["max_abs_diff"]["level"] > 0
    open(bpath, "w").write(orig)
    rows = open(ppath).read().splitlines()
    parts = rows[1].split(",")
    parts[3] = str(int(parts[3]) + 1)
    rows[1] = ",".join(parts)
    open(ppath, "w").write("\n".join(rows) + "\n")
    with pytest.raises(P.PoolError, match="planning CSVs"):
        P.check_csvs(names, pool, str(bpath), str(ppath))


# --- families --------------------------------------------------------------------------------------

def test_keys_normalise():
    assert P.paper_key("https://arxiv.org/pdf/2402.04249v2") == P.paper_key("http://arxiv.org/abs/2402.04249")
    assert P.paper_key("https://openreview.net/forum?id=AbC") == "paper:openreview:AbC"
    assert P.paper_key("https://doi.org/10.1/X") == "paper:doi:10.1/x"
    assert P.src_key("https://www.HuggingFace.co/datasets/Org/name/tree/main") == \
        "src:huggingface.co/datasets/Org"
    assert P.src_key("") is None and P.paper_key(None) is None


def test_families_and_whole_family_exclusion(pool):
    fam = P.families(pool)
    assert fam["benchmarks"] == 9
    assert fam["family_of"]["beta"] == "alpha"                    # shared source key
    assert fam["family_of"]["gamma"] == "delta"                   # arXiv pdf v2 and abs
    assert fam["multi"] == {"alpha": ["alpha", "beta"], "delta": ["delta", "gamma"]}
    assert fam["joining_keys"]["delta"] == ["paper:arxiv:2402.04249"]
    L = P.select(pool)[0]
    assert P.excl_for(["beta"], fam["family_of"], L) == ("alpha", "beta")
    assert P.excl_for(["gamma", "eps"], fam["family_of"], L) == ("delta", "eps", "gamma")
    assert P.excl_for(["lowbet"], fam["family_of"], L) == ("lowbet",)


# --- items -------------------------------------------------------------------------------------------

def test_mode_a_items_and_subjects(tmp_path):
    inter = {("modea-s1", "modea-i3"): "user=sim1", ("modea-s0", "modea-i3"): "user=sim0"}
    sp = bench("modea", n_subjects=3, n_items=90, interactors=lambda s, i: inter.get((s, i)))
    # the first 0/1 row of item 3 is s0's (file order), whose interactor wins benchmark-wide
    root = write_pool(tmp_path / "pool", [sp])
    pairs = P.load_fit_pairs("modea", root=root, content=True)
    by = {p.subject_id: p for p in pairs}
    it = {r.item_key: r.item for r in by["modea-s1"].responses}["modea-i3"]
    assert it == {"item_content": f"{TEXT} modea 3\nA) one\nB) two\nC) three\nD) four",
                  "item_features": "tier=b", "interactors": "user=sim0", "benchmark_id": "modea",
                  "_key": "modea-i3"}
    assert by["modea-s0"].responses[0].item is by["modea-s1"].responses[0].item   # one dict an item
    s = by["modea-s1"].subject
    assert set(s) == set(P.SUBJECT_FIELDS) | {"_sid"} and DISPLAY not in s.values()
    assert s["normalized_name"] == NAMES[1]


def test_a_missing_column_raises(tmp_path):
    root = write_pool(tmp_path / "pool", [bench("nofeat")])
    path = os.path.join(root, "nofeat", "formatted_tables", "items.parquet")
    t = pq.read_table(path).drop(["item_features"])
    pq.write_table(t, path)
    man = json.load(open(os.path.join(root, "manifest.json")))
    man["benchmarks"]["nofeat"]["files"]["items.parquet"] = {
        "bytes": os.path.getsize(path), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"))
    with pytest.raises(P.PoolError, match="no column item_features"):
        P.items_of("nofeat", ["nofeat-i1"], root)


def test_items_of_reads_only_item_columns_in_bounded_batches(tmp_path, monkeypatch):
    root = write_pool(tmp_path / "pool", [bench("big", n_items=200)])
    calls = []
    real_init, real_iter = pq.ParquetFile.__init__, pq.ParquetFile.iter_batches

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        self._spy = (kw.get("buffer_size"), kw.get("pre_buffer"))

    def iter_batches(self, *a, **kw):
        calls.append((kw.get("batch_size"), tuple(kw.get("columns")), self._spy))
        return real_iter(self, *a, **kw)
    monkeypatch.setattr(pq.ParquetFile, "__init__", init)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", iter_batches)
    monkeypatch.setattr(pq, "read_table", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("read_table")))
    want = [f"big-i{i}" for i in (0, 5, 77, 150, 199)]
    got = P.items_of("big", want, root)
    assert calls and all(size <= 256 and cols == P.ITEM_COLUMNS for size, cols, _ in calls)
    assert all(buf and buf > 0 and pre is False for _, _, (buf, pre) in calls)
    monkeypatch.undo()
    assert pq.ParquetFile(os.path.join(root, "big", "formatted_tables", "items.parquet")).metadata.num_row_groups > 3
    whole = pq.read_table(os.path.join(root, "big", "formatted_tables", "items.parquet")).to_pylist()
    ref = {r["item_id"]: (r["content"], r["item_features"]) for r in whole if r["item_id"] in want}
    assert got == ref
    with pytest.raises(P.PoolError, match="not in items.parquet"):
        P.items_of("big", ["big-i0", "nope"], root)


def test_the_fit_does_not_depend_on_item_content(tmp_path):
    root = write_pool(tmp_path / "pool", [bench("cont", n_subjects=4, n_items=90, seed=4)])
    with_text = P.load_fit_pairs("cont", root=root, content=True)
    without = P.load_fit_pairs("cont", root=root, content=False)
    assert any(r.item["item_content"] for r in with_text[0].responses)
    assert all(r.item["item_content"] == "" for p in without for r in p.responses)
    a, b = PR.benchmark_fit(with_text), PR.benchmark_fit(without)
    PR._FITS.clear()
    a2 = PR.benchmark_fit(with_text)
    assert a["t"] == b["t"] == a2["t"] and a["mu"] == b["mu"] and a["item_var"] == b["item_var"]


# --- the evaluation loader -----------------------------------------------------------------------------

def _direct_eval(b, sid, root, keep=None):
    """Every 0/1 response of a pair in file order, straight from the table."""
    t = pq.read_table(os.path.join(root, b, "formatted_tables", "responses.parquet")).to_pylist()
    out = [(r["item_id"], int(r["response"])) for r in t
           if r["subject_id"] == sid and r["response"] in (0.0, 1.0)]
    return [x for x in out if keep is None or x[0] in keep]


def test_eval_loader_equals_a_direct_read(tmp_path):
    rows = []
    rng = np.random.default_rng(0)
    for rep in range(2):                     # interleaved subjects, repeats, nulls, 0.5s
        for s in ("ev-s0", "ev-s1", "ev-s2"):
            for i in range(90):
                v = float(rng.random() < 0.5) if (i + rep) % 11 else (None if i % 2 else 0.5)
                rows.append((s, f"ev-i{i}", v, f"u={s}" if i == 7 else None))
    sp = bench("ev", n_subjects=3, n_items=90, row_group=50)
    sp["rows"] = rows
    root = write_pool(tmp_path / "pool", [sp])
    keep = {"ev-s1": [f"ev-i{i}" for i in range(0, 90, 3)]}
    got = P.load_eval_pairs("ev", ["ev-s1", "ev-s0"], keep=keep, root=root)
    assert [p.subject_id for p in got] == ["ev-s1", "ev-s0"]
    assert [(r.item_key, r.label) for r in got[0].responses] == _direct_eval("ev", "ev-s1", root, set(keep["ev-s1"]))
    assert [(r.item_key, r.label) for r in got[1].responses] == _direct_eval("ev", "ev-s0", root)
    it = {r.item_key: r.item for r in got[1].responses}["ev-i7"]
    first7 = next(r for r in rows if r[1] == "ev-i7" and r[2] in (0.0, 1.0))
    assert it["interactors"] == (first7[3] or "") and it["item_content"].startswith(TEXT)
    # the same pair as the fit loader sees it, first responses only
    fit = {p.subject_id: p for p in P.load_fit_pairs("ev", root=root)}
    firsts = {}
    for r in got[1].responses:
        firsts.setdefault(r.item_key, r.label)
    assert {r.item_key: r.label for r in fit["ev-s0"].responses} == firsts
    assert {r.item_key: r.item for r in fit["ev-s0"].responses}["ev-i7"] == it
    assert os.path.exists(P._eval_index_path("ev"))
    P._RESP.clear()
    again = P.load_eval_pairs("ev", ["ev-s0"], root=root)            # from the cached index
    assert [(r.item_key, r.label) for r in again[0].responses] == _direct_eval("ev", "ev-s0", root)


def test_eval_loader_refuses_ineligible_pairs_and_unbounded_content(tmp_path, monkeypatch):
    sp = bench("ev2", n_subjects=2, n_items=100)
    sp["rows"] += [("ev2-few", f"ev2-i{i}", 1.0, None) for i in range(10)]
    root = write_pool(tmp_path / "pool", [sp])
    with pytest.raises(P.PoolError, match="not an eligible pair"):
        P.load_eval_pairs("ev2", ["ev2-few"], root=root)
    monkeypatch.setattr(P, "MAX_CONTENT_ITEMS", 50)
    with pytest.raises(P.PoolError, match="pass keep"):
        P.load_eval_pairs("ev2", ["ev2-s0"], root=root)
    assert len(P.load_eval_pairs("ev2", ["ev2-s0"], keep={"ev2-s0": ["ev2-i1", "ev2-i2"]},
                                 root=root)[0].item_keys) == 2


def test_the_eval_loader_is_the_only_code_a_fix_may_touch():
    src = inspect.getsource(P)
    tree = ast.parse(src)
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    names |= {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)}
    assert set(P.EVAL_LOADER) <= names
    for node in tree.body:
        name = getattr(node, "name", None) or (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else None)
        if name in P.EVAL_LOADER or name == "main":
            continue
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        assert not used & set(P.EVAL_LOADER), (name, used & set(P.EVAL_LOADER))
    # frozen_ast: blind to an eval-loader edit, not to any other
    for old in ("rows = np.asarray(rows, np.int64)", "ID_CAP = 0.1", '"pool-item", b, x',
                "INDEX_VERSION = 1"):
        assert old in src
    assert P.frozen_ast(src) == P.frozen_ast(src.replace("INDEX_VERSION = 1", "INDEX_VERSION = 2"))
    assert P.frozen_ast(src) != P.frozen_ast(src.replace('"load_eval_pairs", "build_eval_indexes"',
                                                          '"load_eval_pairs", "bundle", "build_eval_indexes"'))
    assert P.frozen_ast(src) == P.frozen_ast(src.replace("rows = np.asarray(rows, np.int64)",
                                                          "rows = np.asarray(rows, dtype=np.int64)"))
    assert P.frozen_ast(src) != P.frozen_ast(src.replace("ID_CAP = 0.1", "ID_CAP = 0.2"))
    assert P.frozen_ast(src) != P.frozen_ast(src.replace('"pool-item", b, x', '"pool-itm", b, x'))
    assert P.frozen_ast(src) == P.frozen_ast(src + "\n# a comment\n")


def test_the_digests_cover_what_they_name():
    mod = inspect.getsource(P)
    defined = {n.name for n in ast.parse(mod).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert set(P.ARITHMETIC) <= defined and set(P.LOADER) <= defined
    allowed = set(P.ARITHMETIC) | {"_text", "PoolError"}
    for name in P.ARITHMETIC + P.LOADER:
        tree = ast.parse(inspect.getsource(getattr(P, name)).strip())
        local = {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)} | \
            {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & defined - local
        ok = allowed if name in P.ARITHMETIC else set(P.LOADER) | {"PoolError", "verify",
                                                                   "_root", "_table", "manifest",
                                                                   "_maxrss_mb"}
        assert used - {name} <= ok, (name, used - ok)
    assert len(P.source_digest("loader")) == len(P.source_digest("arithmetic")) == 64
    assert P.constants_digest() == P.constants_digest()


# --- the bundle arithmetic ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_pairs():
    from tests.synth import make_pairs, make_pair
    pairs = make_pairs(n_subjects=6, n_benchmarks=4, n_items=100, seed=3)
    pairs.append(make_pair("s9", "b9", 100, seed=9))           # a one-pair benchmark
    return pairs


@pytest.fixture(scope="module")
def synth_fits(synth_pairs):
    by = PR._by_benchmark(synth_pairs)
    fits = {b: PR.benchmark_fit(ps) for b, ps in by.items() if len(ps) >= 2}
    T = float(np.median([f["item_var"] for f in fits.values()]))
    fixed = {b: PR.benchmark_fit(ps, s2e=T) for b, ps in by.items()}
    return by, fits, fixed


@pytest.mark.parametrize("excl", [(), ("b0",), ("b1", "b2"), ("b9",)])
def test_fit_mode_equals_prior_build(synth_pairs, synth_fits, excl):
    by, free, fixed = synth_fits
    L = sorted(by)
    fits = {b: free.get(b, fixed[b]) for b in L}
    p1, h1, _ = P.bundle(fits, excl, "fit", L=L, U=L,
                         single=lambda b, T: PR.benchmark_fit(by[b], s2e=T))
    p2, h2 = PR.build(synth_pairs, excl)
    assert P.to_bytes(p1, h1) == P.to_bytes(p2, h2)


def test_literal_mode_is_the_shipped_build_with_level_written_over(synth_pairs, synth_fits):
    from dataclasses import replace
    by, free, fixed = synth_fits
    L = sorted(by)
    p1, h1, _ = P.bundle({b: free.get(b, fixed[b]) for b in L}, (), "literal", L=L, U=L)
    p2, h2 = PR.build(synth_pairs, ())
    assert P.to_bytes(p1, h1) == P.to_bytes(p2, replace(h2, **P.LEVEL_A3))


@pytest.mark.parametrize("X", P.TIE_ORDER)
@pytest.mark.parametrize("excl", [(), ("b1",), ("b0", "b3")])
def test_every_candidate_is_accepted_by_hier(synth_fits, X, excl):
    by, free, fixed = synth_fits
    L, U = sorted(by), [b for b in sorted(by) if b != "b3"]          # b3 stands in for a flagged one
    prior, hyper, rep = P.candidate_bundle(X, excl, free, fixed, L, U, mu0_m=-2.123)
    assert list(hyper.excluded) == sorted(excl) and not set(hyper.included) & set(excl)
    assert list(hyper.included) == [b for b in L if b not in excl]
    assert prior.meta["included"] == list(hyper.included)
    assert prior.meta["excluded"] == list(hyper.excluded)
    assert "b3" not in prior.meta["benchmarks"]
    pr2, hy2 = PR.from_json(json.loads(P.to_bytes(prior, hyper)))
    m = HierPredictor(pr2, hy2)
    subject = {f: "" for f in P.SUBJECT_FIELDS}
    item = {"item_content": "x", "item_features": "", "interactors": "", "benchmark_id": "benchmark_1"}
    p = m.predict([subject, item], [])
    assert isinstance(p, float) and 0 <= p <= 1 and m.failures == 0
    level = {k: getattr(hyper, k) for k in ("mu0", "sigma_mu", "attr_scale")}
    if X == "C1":
        assert level == P.LEVEL_A3
    elif X == "C1m":
        assert level == {"mu0": -2.123, "sigma_mu": 2.5, "attr_scale": 0.5}
    else:
        lev = P.eb_level(fixed, prior, hyper.included)
        assert level == {"mu0": lev["mu0"], "sigma_mu": lev["sigma_mu"], "attr_scale": 0.5}
        assert hyper.sigma_theta == pytest.approx(np.sqrt(np.clip(prior.meta["tau2_res"], .01, .2)))
    assert hyper.id_cap == 0.1


def test_the_archive_literal_rounds_c2s_level(synth_fits):
    by, free, fixed = synth_fits
    L = sorted(by)
    _, h, _ = P.candidate_bundle("C2", (), free, fixed, L, L, literal=True)
    _, h0, _ = P.candidate_bundle("C2", (), free, fixed, L, L)
    assert h.mu0 == round(h0.mu0, 3) and h.sigma_mu == round(h0.sigma_mu, 3)


def test_eb_hand_computed():
    m, s2, it = P.eb([0.0, 1.0, 2.0, 3.0], [0.1] * 4)
    # equal weights: m is the mean, s2 the mean squared deviation less v
    assert m == pytest.approx(1.5) and s2 == pytest.approx(1.25 - 0.1) and it == 2
    x, v = np.array([-2.0, 0.5, 1.0, 4.0, 0.0]), np.array([0.5, 0.1, 0.2, 2.0, 0.05])
    m, s2, _ = P.eb(x, v)
    w = 1 / (s2 + v)
    assert m == pytest.approx(np.sum(w * x) / w.sum(), abs=1e-9)
    assert s2 == pytest.approx(np.sum(w * ((x - m) ** 2 - v)) / w.sum(), abs=1e-8)
    assert P.eb([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])[1] == P.EB_FLOOR


def test_eb_level_drops_wild_levels(synth_fits):
    by, free, fixed = synth_fits
    wild = dict(fixed)
    wild["b2"] = {**fixed["b2"], "mu": 50.0}
    lev = P.eb_level(wild, None, sorted(by))
    assert lev["dropped"] == ["b2"] and lev["n"] == len(by) - 1


def test_b0_centre_and_c1m_matching(synth_fits):
    by, free, fixed = synth_fits
    L = sorted(by)
    prior, hyper, _ = P.candidate_bundle("C1", (), free, fixed, L, L)
    subs = {b: [p.subject for p in ps] for b, ps in by.items()}
    subs["b1"] = [{**s, "release_date": ""} for s in subs["b1"]]
    c, nb = P.b0_centre(prior, hyper, subs)
    assert nb == len(subs) and 0 < c < 1
    assert P.b0_centre(prior, hyper, subs, "undated")[1] == 1
    from dataclasses import replace
    target = P.b0_centre(prior, replace(hyper, mu0=-1.234), subs)[0]
    mu0 = P.match_mu0(prior, hyper, subs, target)
    assert P.b0_centre(prior, replace(hyper, mu0=mu0), subs)[0] == pytest.approx(target, abs=1e-6)
    assert mu0 == pytest.approx(-1.234, abs=1e-5)
    # by hand at one subject: (1 - slip) E sigmoid(mu0 + m + sqrt(S) z)
    s = subs["b0"][0]
    ma, u = prior.attribute(s)
    S = hyper.sigma_mu ** 2 + hyper.sigma_theta ** 2 + u + hyper.sigma_delta ** 2 + \
        hyper.sigma_d ** 2 + hyper.sigma_g ** 2
    z = np.random.default_rng(0).standard_normal(400000)
    mc = (1 - hyper.slip) * np.mean(1 / (1 + np.exp(-(hyper.mu0 + hyper.attr_scale * ma + np.sqrt(S) * z))))
    assert P.b0_centre(prior, hyper, {"b0": [s]})[0] == pytest.approx(mc, abs=3e-3)


def _degenerate_fits(free):
    """The synthetic free fits with b2's made degenerate (amendment 4a): an item variance far
    above the threshold and standings thirty times as wide."""
    bad = dict(free)
    bad["b2"] = {**free["b2"], "item_var": 1000.0, "t": [30.0 * x for x in free["b2"]["t"]],
                 "share": 0.9}
    return bad


def test_the_degenerate_rule_and_its_threshold(synth_fits):
    by, free, fixed = synth_fits
    assert P.DEGENERATE_ITEM_VAR == 50.0 and P.CONSTANTS["DEGENERATE_ITEM_VAR"] == 50.0
    assert P.degenerate(free) == [] and P.degenerate(_degenerate_fits(free)) == ["b2"]
    at = {"x": {**free["b0"], "item_var": P.DEGENERATE_ITEM_VAR}}
    above = {"x": {**free["b0"], "item_var": P.DEGENERATE_ITEM_VAR + 1e-9}}
    assert P.degenerate(at) == [] and P.degenerate(above) == ["x"]
    # a one-pair benchmark has no free fit, so it is never degenerate
    assert P.degenerate({"b9": {**fixed["b9"], "item_var": 1e6}}) == []
    subs = {b: [p.subject for p in ps] for b, ps in by.items()}
    assert set(P.match_set(subs, _degenerate_fits(free))) == set(subs) - {"b2"}
    assert P.match_set(subs, free) == subs


@pytest.mark.parametrize("X", P.TIE_ORDER)
def test_a_degenerate_fit_is_left_out_of_every_pooled_hyperparameter(synth_fits, X):
    """Amendment 4a: T, the group share, sigma_delta and sigma_attr, and C2's level leave
    b2 out; its rows still make the subject prior (the ridge and the identity table) and it
    stays in `included`; sigma_theta is the prior's own clipped tau2_res. Amendment 4b: the
    ridge's s2_res, and so the coefficient covariance and every u, leave it out too."""
    by, free, fixed = synth_fits
    L = sorted(by)
    bad = _degenerate_fits(free)
    prior, hyper, rep = P.candidate_bundle(X, (), bad, fixed, L, L, mu0_m=-2.1)
    assert rep["degenerate"] == ["b2"] and "b2" in hyper.included and "b2" in prior.meta["benchmarks"]
    others = [bad[b] for b in sorted(free) if b != "b2"]
    T = float(np.median([f["item_var"] for f in others]))
    shares = [f["share"] for f in others if f["share"] is not None]
    share = min(float(np.median(shares)), PR.G_CAP) if shares else None
    assert hyper.sigma_d ** 2 + hyper.sigma_g ** 2 == pytest.approx(T)
    if share is not None:
        assert hyper.sigma_g ** 2 == pytest.approx(share * T)
    rows = P.standings_rows({b: bad.get(b, fixed[b]) for b in L}, L)
    assert {r[0] for r in rows} >= {"b2"}
    full = PR.from_standings(rows, P.ALPHA, L, (), None)
    sub = PR.from_standings([r for r in rows if r[0] != "b2"], P.ALPHA).meta
    # the prior is from_standings on every row, b2's included (the standings, the identity
    # table, s2d and tau2_res), except, by amendment 4b, its coefficient covariance, taken at
    # the s2_res pooled without b2, and the pooled variances of its meta
    got, want = PR.to_json(prior, hyper)["prior"], PR.to_json(full, hyper)["prior"]
    assert {k: v for k, v in got.items() if k not in ("cov", "meta")} == \
        {k: v for k, v in want.items() if k not in ("cov", "meta")}
    pooled = ("s2_res", "u_mean", "total_var")
    assert {k: v for k, v in got["meta"].items() if k not in pooled} == \
        {k: v for k, v in want["meta"].items() if k not in pooled}
    assert {k: got["meta"][k] for k in pooled} == {k: sub[k] for k in pooled}
    _, _, X_, Ai = PR._ridge(rows, P.ALPHA)
    assert np.array_equal(prior.cov, max(sub["s2_res"], 1e-6) * Ai @ X_.T @ X_ @ Ai)
    assert np.allclose(prior.cov, full.cov * sub["s2_res"] / full.meta["s2_res"], rtol=1e-12, atol=0)
    u_full = np.einsum("ij,jk,ik->i", X_, full.cov, X_).mean()
    u_got = np.einsum("ij,jk,ik->i", X_, prior.cov, X_).mean()
    assert u_got == pytest.approx(rep["attribute_cov"]["u_mean_rows"])
    assert rep["attribute_cov"]["s2_res_all_rows"] == full.meta["s2_res"] > 2 * sub["s2_res"]
    assert u_full > 2 * u_got          # kept, the degenerate fit would have inflated every u
    # fit_hyper's arithmetic on the bundle's own prior meta gives the bundle's widths
    hy2, _ = P.hyper_arith({b: bad.get(b, fixed[b]) for b in L}, L, prior, (), deg=("b2",))
    assert (hy2.sigma_delta, hy2.sigma_attr) == (hyper.sigma_delta, hyper.sigma_attr)
    th2 = float(np.clip(full.meta["tau2_res"], *P.ID_CLIP))
    assert hyper.sigma_theta == math.sqrt(th2)
    assert hyper.sigma_delta == math.sqrt(max(sub["s2_res"] - sub["u_mean"] - th2, 0.1)) * PR.WIDEN[1]
    assert hyper.sigma_attr == math.sqrt(max(sub["total_var"] - sub["s2_res"], 0.0))
    assert rep["pooled"]["s2_res"] == sub["s2_res"] and rep["pooled"]["pooled_rows"] == len(rows) - 6
    # kept, the degenerate fit would have widened sigma_delta
    kept = math.sqrt(max(full.meta["s2_res"] - full.meta["u_mean"] - th2, 0.1)) * PR.WIDEN[1]
    assert kept > 2 * hyper.sigma_delta
    if X == "C2":
        lev = P.eb_level(fixed, prior, [b for b in L if b != "b2"])
        assert (hyper.mu0, hyper.sigma_mu) == (lev["mu0"], lev["sigma_mu"])
        assert lev["n"] == len(L) - 1
    # the same bundle without the degenerate fit's distortion is untouched: no degenerate
    # fit, no change (the rule is a no-op on the synthetic and the legacy fits)
    p0, h0, r0 = P.candidate_bundle(X, (), free, fixed, L, L, mu0_m=-2.1)
    assert r0["degenerate"] == [] and "pooled" not in r0 and "attribute_cov" not in r0


def test_pooled_prior_rescales_the_covariance_only(synth_fits):
    """Amendment 4b (pooled_prior): with nothing pooled the prior comes back as it was; a
    pooled s2_res rescales the coefficient covariance and so u = x' C x, and nothing else of
    what a subject's attribute prediction reads; rows that are not the prior's are refused."""
    by, free, fixed = synth_fits
    L = sorted(by)
    bad = _degenerate_fits(free)
    rows = P.standings_rows({b: bad.get(b, fixed[b]) for b in L}, L)
    full = PR.from_standings(rows, P.ALPHA, L, (), None)
    assert P.pooled_prior(rows, full, None) == (full, None)
    assert P.pooled_prior(rows, full, dict(full.meta)) == (full, None)   # no pooled_rows
    assert P.pooled_prior(rows, None, None) == (None, None)
    meta = P.pooled_meta(rows, full, ["b2"])
    got, rep = P.pooled_prior(rows, full, meta)
    assert rep["pooled_rows"] == meta["pooled_rows"] == len(rows) - 6 and rep["rows"] == len(rows)
    assert "pooled_rows" not in got.meta and set(got.meta) == set(full.meta)
    assert got.meta["s2_res"] == meta["s2_res"] < full.meta["s2_res"]
    assert full.meta["s2_res"] == PR.from_standings(rows, P.ALPHA, L, (), None).meta["s2_res"]
    for r in rows[:12]:
        s = r[1]
        m0, u0 = full.attribute(s)
        m1, u1 = got.attribute(s)
        assert m1 == m0 and u1 == pytest.approx(u0 * meta["s2_res"] / full.meta["s2_res"], rel=1e-9)
    # the bundle's bytes round-trip with the rescaled covariance
    _, hyper, _ = P.candidate_bundle("C1", (), bad, fixed, L, L)
    back, _ = PR.from_json(json.loads(P.to_bytes(got, hyper)))
    assert np.array_equal(back.cov, got.cov)
    with pytest.raises(P.PoolError, match="not the prior's"):
        P.pooled_prior(rows[:-3], full, meta)


def test_fit_mode_keeps_fit_hypers_arithmetic_with_a_degenerate_fit(synth_pairs, synth_fits):
    """RP4's "fit" mode is paiec.prior.fit_hyper's arithmetic: amendment 4a does not apply."""
    by, free, fixed = synth_fits
    L = sorted(by)
    bad = _degenerate_fits(free)
    fits = {b: bad.get(b, fixed[b]) for b in L}
    _, h, rep = P.bundle(fits, (), "fit", L=L, U=L,
                         single=lambda b, T: PR.benchmark_fit(by[b], s2e=T))
    assert rep["degenerate"] == [] and "pooled" not in rep
    T = float(np.median([bad[b]["item_var"] for b in free]))
    assert h.sigma_d ** 2 + h.sigma_g ** 2 == pytest.approx(T)


def test_fit_all_leaves_degenerate_fits_out_of_T_full(pool, monkeypatch):
    L, U, _ = P.select(pool)
    real = P._run_fits

    def inflated(tasks, *a, **k):
        out = real(tasks, *a, **k)
        for (b, s2e), f in list(out.items()):
            if b == "alpha" and s2e is None:
                out[(b, s2e)] = {**f, "item_var": 999.0}
        return out
    monkeypatch.setattr(P, "_run_fits", inflated)
    free, fixed, T = P.fit_all(L, 1, pool, log=lambda *_: None, fresh=False)
    assert P.degenerate(free) == ["alpha"]
    assert T == float(np.median([free[b]["item_var"] for b in free if b != "alpha"]))
    assert all(fixed[b]["s2e_given"] == T for b in L)


def test_candidates_table():
    assert P.CANDIDATES == {"C1": "literal", "C1m": "matched", "C2": "eb"}
    assert P.TIE_ORDER == ("C1", "C1m", "C2")
    assert P.LEVEL_A3 == {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
    assert P.MATCHED_SIGMA_MU == 2.5 and P.ATTR_SCALE == 0.5 and P.ID_CAP == 0.1


def test_level_a3_is_archive_3s():
    try:
        path = P.a3_prior_path()
    except P.PoolError:
        pytest.skip("no copy of archive-3's prior.json")
    _, h = PR.load(path)
    assert {k: getattr(h, k) for k in P.LEVEL_A3} == P.LEVEL_A3


# --- fits, cache, lists, byte stability --------------------------------------------------------------------

def test_fit_all_caches_and_indexes(pool):
    L, U, _ = P.select(pool)
    free, fixed, T = P.fit_all(L, 1, pool, log=lambda *_: None, fresh=False)
    assert set(fixed) == set(L) and "eps" not in free and set(free) == set(L) - {"eps"}
    assert T == float(np.median([free[b]["item_var"] for b in free]))
    assert all(fixed[b]["s2e_given"] == T for b in L)
    f2, x2, T2 = P.load_fits(pool)
    assert (f2, x2, T2) == (free, fixed, T)
    again = P.fit_all(L, 1, pool, log=lambda *_: None, fresh=False)        # from the cache
    assert again[0]["alpha"]["fit_cpu"] == free["alpha"]["fit_cpu"]
    entry = json.load(open(os.path.join(P.FITS_DIR, "index.json")))["entries"]["alpha/free"]
    open(os.path.join(P.FITS_DIR, entry["file"]), "a").write(" ")
    with pytest.raises(P.PoolError, match="changed"):
        P.load_fits(pool)
    for b in L:
        assert set(free.get(b, fixed[b])) >= set(P.FIT_KEEP)


def test_lists_are_names_and_keys_and_byte_stable(pool, tmp_path):
    out = tmp_path / "tools"
    d1 = P.write_lists(pool, str(out))
    d2 = P.write_lists(pool, str(out))
    assert d1 == d2
    bench_, fams = P.read_lists(str(out))
    assert bench_["L"] == P.select(pool)[0] and bench_["E"] == bench_["L"]   # no legacy here
    assert bench_["revision"] == P.REVISION and bench_["flagged"] == list(P.FLAGGED)
    assert fams["family_of"]["beta"] == "alpha"
    raw = (out / "pool_benchmarks.json").read_bytes() + (out / "pool_families.json").read_bytes()
    assert TEXT.encode() not in raw and DISPLAY.encode() not in raw


SCRIPT = r"""
import hashlib, json, sys
from paiec import pool as P
root, fits = sys.argv[1], sys.argv[2]
P.FITS_DIR, P.EVAL_DIR = fits, fits + "_eval"
L, U, _ = P.select(root)
free, fixed, T = P.fit_all(L, 1, root, log=lambda *_: None, fresh=False)
out = {}
for X in P.TIE_ORDER:
    for excl in ((), ("alpha", "beta")):
        p, h, _ = P.candidate_bundle(X, excl, free, fixed, L, U, mu0_m=-2.0, literal=not excl)
        out[f"{X}{excl}"] = hashlib.sha256(P.to_bytes(p, h)).hexdigest()
print(json.dumps(out, sort_keys=True))
"""


def test_bundles_are_byte_stable_across_processes_and_hash_seeds(pool, tmp_path):
    got = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "OPENBLAS_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        env.pop("PYTHONPATH", None)
        r = subprocess.run([sys.executable, "-c", SCRIPT, pool, str(tmp_path / f"fits{seed}")],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        got.append(r.stdout.strip().splitlines()[-1])
    assert got[0] == got[1]


def test_subject_sets_shift_dates(pool):
    real, ds = P.subject_sets(["alpha"], pool)
    from paiec.testlike import shift_date
    for a, b in zip(real["alpha"], ds["alpha"]):
        assert b["release_date"] == (shift_date(a["release_date"], 1.25) if a["release_date"] else "")
        assert {k: v for k, v in a.items() if k not in ("release_date", "access_date")} == \
            {k: v for k, v in b.items() if k not in ("release_date", "access_date")}


def test_scales_of_mixed_counts_only(pool):
    got = P.scales_of_mixed(["mixok"], pool)
    assert got == {"mixok": {"items": 100, "scales": {"discrete:2": 100}, "declared_01": 100,
                             "unparsed": 0}}


def test_the_catalogue_mcq_flag(tmp_path):
    root = write_pool(tmp_path / "pool", [bench("q1", description="A Multiple-choice exam"),
                                          bench("q2", description="free-form MCQs"),
                                          bench("q3", description="open answers")])
    assert [P.mcq_flag(b, root) for b in ("q1", "q2", "q3")] == [True, True, False]


# --- RP4 on the legacy five (data/<b>/, when present) ----------------------------------------------------

def _legacy_present():
    from paiec.data import DATA_DIR, BINARY
    return all(os.path.exists(os.path.join(DATA_DIR, b, "response.parquet")) for b in BINARY)


@pytest.fixture(scope="module")
def legacy():
    if not _legacy_present():
        pytest.skip("legacy data not present")
    from paiec.data import load_pairs
    from paiec.official import eligible
    pairs = eligible(load_pairs())
    by = PR._by_benchmark(pairs)
    free = {b: PR.benchmark_fit(ps) for b, ps in by.items() if len(ps) >= 2}
    T = float(np.median([f["item_var"] for f in free.values()]))
    fits = {b: free.get(b) or PR.benchmark_fit(ps, s2e=T) for b, ps in by.items()}
    return pairs, by, fits


@pytest.mark.parametrize("parent", P.LEGACY)
def test_rp4_leave_one_out_equals_prior_build(legacy, parent):
    pairs, by, fits = legacy
    L = sorted(by)
    p1, h1, _ = P.bundle(fits, (parent,), "fit", L=L, U=L,
                         single=lambda b, T: PR.benchmark_fit(by[b], s2e=T))
    p2, h2 = PR.build(pairs, (parent,))
    assert P.to_bytes(p1, h1) == P.to_bytes(p2, h2)


def test_rp4_excl_empty_with_level_written_over_is_the_shipped_build(legacy):
    from dataclasses import replace
    pairs, by, fits = legacy
    L = sorted(by)
    p1, h1, _ = P.bundle(fits, (), "literal", L=L, U=L)
    p2, h2 = PR.build(pairs, ())
    assert P.to_bytes(p1, h1) == P.to_bytes(p2, replace(h2, **P.LEVEL_A3))


def test_rp4_archive_3s_prior_json_byte_for_byte(legacy):
    """Under the BLAS threading archive-3 was built with (OpenBLAS's default),
    the literal bundle is archive-3's prior.json; under another (one thread,
    as the lock runs) neither it nor paiec.prior.build reproduces those last
    bits, and this check is the reproduce stage's, run in the build's setting."""
    try:
        a3 = P.a3_prior_path()
    except P.PoolError:
        pytest.skip("no copy of archive-3's prior.json")
    pairs, _, _ = legacy
    got = P.rp4_legacy(pairs, a3)
    assert got["literal_equals_shipped_build"] and all(got["leave_one_out"].values())
    if not got["shipped_build_equals_archive_3"]:
        pytest.skip(f"this BLAS setting ({got['blas']}) does not reproduce archive-3's bytes")
    assert got["literal_equals_archive_3"]


def test_the_fits_stage_runs_end_to_end_on_a_synthetic_pool(pool, tmp_path, monkeypatch):
    """The stage's checks against the real planning files are stubbed; everything
    it computes runs, and its record holds no item text and no score."""
    monkeypatch.setattr(P, "MANIFEST_SHA256", P.manifest_sha256(pool))
    monkeypatch.setattr(P, "FLAGGED", ("lowbet",))
    monkeypatch.setattr(P, "check_selection_csv", lambda L: {"stub": True})
    monkeypatch.setattr(P, "check_csvs", lambda names, root=None: {"stub": True})
    monkeypatch.setattr(P, "check_families", lambda fam: {"stub": True})
    for v in ("PYTHONHASHSEED",) + P.BLAS_VARS:
        monkeypatch.setenv(v, "0" if v == "PYTHONHASHSEED" else "1")
    real_select = P.select
    monkeypatch.setattr(P, "select", lambda root=None, jobs=1, flagged=None:
                        real_select(root, jobs, ("lowbet",)))
    try:
        P.a3_prior_path()
    except P.PoolError:
        pytest.skip("no copy of archive-3's prior.json")
    out = tmp_path / "pool_fits.json"
    rec = P.fits_stage(1, pool, str(out), floor=True, probes=(), log=lambda *_: None)
    assert rec["selection"]["L"] == 7 and rec["selection"]["U"] == 6
    assert set(rec["bundles"]) == {"C1", "C1m", "C2"}
    for X, b in rec["bundles"].items():
        p = os.path.join(P.EVAL_DIR, "bundles", f"archive_{X}.json")
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == b["sha256"]
    assert rec["bundles"]["C1m"]["level"]["mu0"] == rec["mu0_m"]["locked"] == round(rec["mu0_m"]["raw"], 3)
    assert rec["b0_centre_minus_A3"]["C1m"]["plus_1.25y:all"] == pytest.approx(0, abs=2e-3)
    # amendment 4a: no degenerate fit here, so the matching set is E and T_full is the median
    assert rec["degenerate"] == [] and rec["degenerate_rule"]["threshold"] == 50.0
    assert rec["mu0_m"]["matching_set"]["left_out"] == []
    # amendment 4b: nothing pooled, so the covariance is the all-row prior's
    assert rec["pooled"]["attribute_cov"] is None and rec["u"]["E_pair_subjects"]["all_rows_scaled"] is None
    assert rec["u"]["E_pair_subjects"]["pairs"] == rec["b0_pairs"]["E"] and rec["u"]["E_pair_subjects"]["pooled"] >= 0
    assert rec["b0_centre_matching_set_minus_A3"]["C1m"]["plus_1.25y:all"] == pytest.approx(0, abs=2e-3)
    assert rec["C2"]["literal"]["mu0"] == round(rec["C2"]["mu0"], 3)
    raw = out.read_bytes()
    assert TEXT.encode() not in raw and DISPLAY.encode() not in raw
    assert b"brier" not in raw.lower() and b'"alc' not in raw.lower()
    assert set(rec["floor_share"]) == set(P.evaluation_set(P.select(pool)[0]))
