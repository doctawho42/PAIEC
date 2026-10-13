"""The expanded measurement-db pool, offline: files, selection, families, pairs,
per-benchmark fits and the candidate bundles of amendment 4.

docs/plans/p2_pool_eval.md ("PE") is the design; section numbers below are
its. Nothing here ships: the module is not in tools/build_submission.PACKAGE
and no shipped module imports it. It reads measurement-db at REVISION from
data/pool/<b>/formatted_tables/ (tools/fetch_pool.py --items), and every pool
file is checked against data/pool/manifest.json (size and sha256) before a
process first reads it (verify).

  * Selection (PE 3.3): L, the level part, is every public item-level
    benchmark whose responses are binary, or mixed with a 0/1 share of at
    least SHARE01_MIN among non-null responses, with at least one eligible
    pair; U, the subject part, is L less the nine direction-flagged
    benchmarks; E, the evaluation set, is L less the legacy five.
  * Eligibility as paiec/data.py and paiec/official.py apply it: only 0.0 and
    1.0 responses count (nulls and every other value dropped), a pair is
    (subject_id, benchmark) with at least MIN_ITEMS distinct items, and an
    item's label is its first 0/1 response in file order.
  * Families (PE 3.5): catalogue entries joined by a shared source_url key or
    paper_url key; a family is a connected component; excl_for gives the
    whole families of a run's benchmarks.
  * Items, Mode A (PE 3.2): content and item_features from items.parquet,
    read with column projection (never verifier or grading_criterion) and in
    bounded batches; interactors from the item's first 0/1 response row,
    benchmark-wide. Item text is held in memory only and never written: the
    caches here hold ids, offsets, counts and fitted numbers.
  * Fits (PE 3.4, 4.1): paiec.prior.benchmark_fit per benchmark, at most K
    pairs (round robin over canonical names) and M_P items a pair (one
    benchmark-wide hash order), in worker processes with one BLAS thread,
    cached under data/pool_fits/ by revision, benchmark, caps, item variance
    and the loader's source digest.
  * Bundles (PE 4.1-4.3): the subject prior from U's standings, a copy of
    paiec.prior.fit_hyper's arithmetic (checked against fit_hyper on the
    legacy five, RP4), and the level of each candidate in CANDIDATES: C1
    archive-3's LEVEL, C1m LEVEL with mu0 matched to archive-3's B0 centre at
    +DATE_SHIFT years, C2 the pool's empirical-Bayes level. "fit" keeps
    fit_hyper's own levels (RP4 only). Amendment 4a: a benchmark whose free
    fit has an item variance above DEGENERATE_ITEM_VAR (degenerate()) is left
    out of every pooled hyperparameter of every candidate (T and T_full, the
    group share, sigma_delta and sigma_attr, C2's level distribution, C1m's
    matching); its pairs still serve the standings and the identity table.
    Amendment 4b: the attribute ridge's residual variance s2_res, and so each
    subject's attribute variance u, is one of them (pooled_prior).
  * The evaluation loader (PE 5.2): load_eval_pairs reads a drawn pair's
    0/1 responses through its own index (ids and offsets in
    data/pool_eval/index/), only the row groups that hold them, and the
    content of the kept items only.

N5 (PE 9 item 11): a post-lock fix may change only the EVAL_LOADER
definitions; frozen_ast() is the rest of this file as the lock compares it.
Nothing outside EVAL_LOADER uses them but main(). The fits cache is keyed on
source_digest("loader"); source_digest("arithmetic") and constants_digest()
are recorded beside the file's own sha256.

    python -m paiec.pool                     # report on the cached fits (like prior.main)
    python -m paiec.pool --check             # manifest, selection, families, CSV agreement
    python -m paiec.pool --lists             # tools/pool_benchmarks.json, tools/pool_families.json
    python -m paiec.pool --index --jobs 4    # the evaluation loader's index for E
    PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        VECLIB_MAXIMUM_THREADS=1 python -m paiec.pool --fit --jobs 4   # the fits stage
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import resource
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

import numpy as np

from paiec import prior as PR
from paiec.data import SUBJECT_FIELDS
from paiec.evaluator import Pair, Response, stable_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the pool's files; PAIEC_POOL overrides
POOL = os.environ.get("PAIEC_POOL") or os.path.join(ROOT, "data", "pool")
#: fits and pair-table caches (no item text)
FITS_DIR = os.path.join(ROOT, "data", "pool_fits")
#: evaluation-side caches: the eval index (ids and offsets), the bundles, a3_prior.json
EVAL_DIR = os.path.join(ROOT, "data", "pool_eval")
TOOLS = os.path.join(ROOT, "tools")
PLANNING = os.path.join(ROOT, "data", "p2_work", "pool_design")

# --- constants (PE 3, 4; every one is in CONSTANTS and constants_digest) -------------

REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"
#: data/pool/manifest.json as fetched on 2026-10-08 (PE 3.1); the fits stage requires it
MANIFEST_SHA256 = "ff5482069780ea7844170f2f4c6596b7cda3b3aa568bbbba3d9e4fede9dae915"
#: archive-3's prior.json (PE 2, 4.6)
A3_PRIOR_SHA256 = "c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770"
LEGACY = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
#: direction-flagged: lower_is_better or unordered (PE 3.3); in L, not in U
FLAGGED = ("alignment_faking", "crow", "emoji_attack", "harmbench", "jailbreakbench",
           "llmail_inject", "safeagentbench", "tensortrust", "wmt_mqm")
FLAG_DIRECTIONS = ("lower_is_better", "unordered")
KINDS = ("binary", "mixed")
SHARE01_MIN = 0.95
MIN_ITEMS = 80
K = 100
M_P = 2000
DATE_SHIFT = 1.25
MCQ_RE = re.compile(r"\b(multiple|multi)[\s-]*choice\b|\bmcqa?s?\b", re.I)
MCQ_FIELDS = ("name", "one_line_description", "description", "benchmark_features")
CATALOGUE_COLUMNS = ("benchmark_id", "name", "release", "granularity", "response_type",
                     "response_scale", "source_url", "paper_url") + MCQ_FIELDS[1:]
RESPONSE_COLUMNS = ("subject_id", "item_id", "interactors", "response")
#: the only item columns ever read (N8): never verifier or grading_criterion
ITEM_COLUMNS = ("item_id", "content", "item_features")
ALPHA = 2.0
ID_CLIP = (0.01, 0.2)
ID_CAP = 0.1
#: archive-3's LEVEL (submission/model.py at a725df9), C1's level
LEVEL_A3 = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
MATCHED_SIGMA_MU = 2.5
ATTR_SCALE = 0.5
LEVEL_DROP = 10.0
EB_MAX_ITER = 500
EB_TOL = 1e-10
EB_FLOOR = 1e-6
MATCH_BRACKET = (-10.0, 10.0)
MATCH_TOL = 1e-6
LITERAL_DECIMALS = 3
GH_POINTS = 80
#: amendment 4a: a free fit whose item variance is above this is degenerate and left out
#: of every pooled hyperparameter (degenerate()); PE 9 item 3's descriptive report of
#: item variances above REPORT_ITEM_VAR is kept beside it
DEGENERATE_ITEM_VAR = 50.0
REPORT_ITEM_VAR = 100.0
#: the largest difference allowed between the levels and logits derived here and the
#: planning CSVs (impl_plan 2; experiments/pool_eval.CSV_TOL is the same)
CSV_TOL = 1e-12
CANDIDATES = {"C1": "literal", "C1m": "matched", "C2": "eb"}
#: order of preference for ties (PE 4.3)
TIE_ORDER = ("C1", "C1m", "C2")
LEVEL_MODES = ("literal", "matched", "eb", "fit")
#: a load_eval_pairs call may read the content of at most this many items (N8)
MAX_CONTENT_ITEMS = 4000
#: items_of reads at most ITEM_BATCH rows at a time through a READ_BUFFER-byte stream (N8)
ITEM_BATCH = 64
READ_BUFFER = 1 << 20
RESP_CACHE = 2
BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS")
#: the planning counts the fits stage re-measures (PE 3.3, 3.5)
EXPECTED = {"L": 137, "U": 128, "E": 132, "families": 230, "pairs_E": 12668, "pairs_U": 8388}
SELECTION_RULE = (
    f"L: catalogue status ok, release public and granularity item at revision {REVISION}; "
    "response_type binary, or mixed with a share of 0.0/1.0 among non-null responses of at "
    f"least {SHARE01_MIN}; at least one eligible pair, a (subject_id, benchmark) with at least "
    f"{MIN_ITEMS} distinct items carrying a 0/1 response. U: L less the direction-flagged "
    f"benchmarks (response_scale.direction {' or '.join(FLAG_DIRECTIONS)}). E: L less the "
    "legacy five.")
FAMILY_RULE = ("two catalogue entries are joined when they share a source_url key (host "
               "lower-cased without www., plus the first two path segments) or a paper_url "
               "key (arXiv id without version, OpenReview forum id, DOI, else host plus path); "
               "a family is a connected component, named by its alphabetically first member")
CONSTANTS = dict(
    REVISION=REVISION, MANIFEST_SHA256=MANIFEST_SHA256, A3_PRIOR_SHA256=A3_PRIOR_SHA256,
    LEGACY=LEGACY, FLAGGED=FLAGGED, FLAG_DIRECTIONS=FLAG_DIRECTIONS, KINDS=KINDS,
    SHARE01_MIN=SHARE01_MIN, MIN_ITEMS=MIN_ITEMS, K=K, M_P=M_P, DATE_SHIFT=DATE_SHIFT,
    MCQ_RE=MCQ_RE.pattern, MCQ_FIELDS=MCQ_FIELDS, CATALOGUE_COLUMNS=CATALOGUE_COLUMNS,
    RESPONSE_COLUMNS=RESPONSE_COLUMNS, ITEM_COLUMNS=ITEM_COLUMNS, ALPHA=ALPHA, ID_CLIP=ID_CLIP,
    ID_CAP=ID_CAP, LEVEL_A3=LEVEL_A3, MATCHED_SIGMA_MU=MATCHED_SIGMA_MU, ATTR_SCALE=ATTR_SCALE,
    LEVEL_DROP=LEVEL_DROP, EB_MAX_ITER=EB_MAX_ITER, EB_TOL=EB_TOL, EB_FLOOR=EB_FLOOR,
    MATCH_BRACKET=MATCH_BRACKET, MATCH_TOL=MATCH_TOL, LITERAL_DECIMALS=LITERAL_DECIMALS,
    GH_POINTS=GH_POINTS, DEGENERATE_ITEM_VAR=DEGENERATE_ITEM_VAR,
    REPORT_ITEM_VAR=REPORT_ITEM_VAR, CSV_TOL=CSV_TOL, CANDIDATES=CANDIDATES,
    TIE_ORDER=TIE_ORDER, LEVEL_MODES=LEVEL_MODES, MAX_CONTENT_ITEMS=MAX_CONTENT_ITEMS,
    ITEM_BATCH=ITEM_BATCH, READ_BUFFER=READ_BUFFER,
    EXPECTED=EXPECTED, SELECTION_RULE=SELECTION_RULE, FAMILY_RULE=FAMILY_RULE)


class PoolError(RuntimeError):
    """A pool file, a rule or a check failed: the stage that meets it stops."""


def _text(v) -> str:
    """'' for a missing value, else str() (paiec.data._clean without pandas)."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _json_bytes(obj) -> bytes:
    """Canonical JSON for the files this module writes: sorted keys, one space."""
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def _write(path, data: bytes):
    """Write through a temporary file, so a reader never sees half a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10


def _root(root=None) -> str:
    return os.path.abspath(root or POOL)


def _table(t: str) -> str:
    return t if t.endswith(".parquet") else f"{t}.parquet"


# --- the manifest and the files (PE 3.1) ----------------------------------------------

_MANIFESTS: dict = {}
_VERIFIED: dict = {}


def manifest(root=None) -> dict:
    """data/pool/manifest.json, refused unless it is at REVISION."""
    path = os.path.join(_root(root), "manifest.json")
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise PoolError(f"{path}: no manifest") from None
    stamp = (st.st_size, st.st_mtime_ns)
    got = _MANIFESTS.get(path)
    if got is None or got[0] != stamp:
        with open(path, "rb") as f:
            raw = f.read()
        m = json.loads(raw)
        if m.get("revision") != REVISION:
            raise PoolError(f"{path} is at revision {m.get('revision')}, not {REVISION}")
        got = (stamp, m, hashlib.sha256(raw).hexdigest())
        _MANIFESTS[path] = got
    return got[1]


def manifest_sha256(root=None) -> str:
    manifest(root)
    return _MANIFESTS[os.path.join(_root(root), "manifest.json")][2]


def verify(b, table, root=None) -> str:
    """The path of one pool file after checking its size and sha256 against the
    manifest, once per process and file (again if the file changes on disk).
    Raises PoolError on a missing entry, a missing file or a mismatch."""
    root = _root(root)
    table = _table(table)
    entry = manifest(root)["benchmarks"].get(b, {}).get("files", {}).get(table)
    if entry is None:
        raise PoolError(f"{b}/{table}: not in the manifest")
    path = os.path.join(root, b, "formatted_tables", table)
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise PoolError(f"{b}/{table}: missing on disk") from None
    stamp = (entry["sha256"], st.st_size, st.st_mtime_ns)
    if _VERIFIED.get(path) == stamp:
        return path
    if st.st_size != entry["bytes"]:
        raise PoolError(f"{b}/{table}: {st.st_size} bytes, the manifest says {entry['bytes']}")
    if _sha256_file(path) != entry["sha256"]:
        raise PoolError(f"{b}/{table}: sha256 differs from the manifest")
    _VERIFIED[path] = stamp
    return path


def file_sha256(b, table, root=None) -> str:
    """The manifest's sha256 of one pool file, after verify."""
    verify(b, table, root)
    return manifest(root)["benchmarks"][b]["files"][_table(table)]["sha256"]


def verify_all(names, root=None) -> dict:
    """Every manifest-listed file of these benchmarks; a count and bytes."""
    n = size = 0
    for b in sorted(names):
        for table, e in sorted(manifest(root)["benchmarks"][b].get("files", {}).items()):
            verify(b, table, root)
            n += 1
            size += e["bytes"]
    return {"benchmarks": len(names), "files": n, "bytes": size}


def _require(path, cols):
    """Raise PoolError unless the parquet file has every one of `cols`."""
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    missing = [c for c in cols if c not in names]
    if missing:
        raise PoolError(f"{path}: no column {', '.join(missing)}")


# --- the catalogue and the selection (PE 3.3) -----------------------------------------

def catalogue(b, root=None) -> dict:
    """One benchmark's catalogue row (benchmarks.parquet), its response_scale
    parsed; release, granularity, response_type and response_scale must agree
    with the manifest's copy."""
    import pyarrow.parquet as pq
    path = verify(b, "benchmarks.parquet", root)
    _require(path, CATALOGUE_COLUMNS)
    rows = pq.read_table(path, columns=list(CATALOGUE_COLUMNS)).to_pylist()
    if len(rows) != 1:
        raise PoolError(f"{b}: {len(rows)} catalogue rows")
    row = rows[0]
    entry = manifest(root)["benchmarks"][b]
    for k in ("release", "granularity", "response_type", "response_scale"):
        if _text(entry.get(k)) != _text(row.get(k)):
            raise PoolError(f"{b}: catalogue {k} {row.get(k)!r}, manifest {entry.get(k)!r}")
    try:
        scale = json.loads(row["response_scale"]) if row.get("response_scale") else {}
    except ValueError:
        raise PoolError(f"{b}: response_scale is not JSON") from None
    out = {k: row.get(k) for k in CATALOGUE_COLUMNS}
    out.update(status=entry.get("status"), direction=(scale or {}).get("direction"),
               scale_kind=(scale or {}).get("kind"))
    return out


def mcq_flag(b, root=None) -> bool:
    """The catalogue multiple-choice flag (PE 3.7): MCQ_RE on the catalogue's
    name and descriptions, never on item text."""
    row = catalogue(b, root)
    return bool(MCQ_RE.search(" ".join(_text(row.get(k)) for k in MCQ_FIELDS)))


class _Resp:
    """One benchmark's 0/1 responses in file order, as codes into its subject and
    item id lists (no item text). pos are the rows' positions in the file."""
    __slots__ = ("b", "sids", "iids", "pos", "s", "i", "y", "inames", "ic",
                 "rows", "nonnull", "n01", "_first", "_ifirst")

    def first(self):
        """Indices into the 0/1 rows of each (subject, item) cell's first
        response, in file order (np.unique's return_index is the first
        occurrence: it sorts stably)."""
        if self._first is None:
            key = self.s.astype(np.int64) * (len(self.iids) + 1) + self.i
            _, idx = np.unique(key, return_index=True)
            self._first = np.sort(idx)
        return self._first

    def item_first_row(self):
        """{item code: index into the 0/1 rows of the item's first response}."""
        if self._ifirst is None:
            _, idx = np.unique(self.i, return_index=True)
            self._ifirst = dict(zip(self.i[idx].tolist(), idx.tolist()))
        return self._ifirst

    def cells_by_subject(self):
        """(fs, fi, fy, groups): the first cells' subject codes, item codes and
        labels, and {subject code: indices into them in file order}."""
        f = self.first()
        fs, fi, fy = self.s[f], self.i[f], self.y[f]
        order = np.argsort(fs, kind="stable")
        counts = np.bincount(fs, minlength=len(self.sids))
        ends = np.cumsum(counts)
        groups = {c: order[ends[c] - counts[c]:ends[c]] for c in np.flatnonzero(counts).tolist()}
        return fs, fi, fy, groups


def _read_responses(path, b) -> _Resp:
    """Named columns only; 0.0 and 1.0 kept, nulls and other values dropped."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    _require(path, RESPONSE_COLUMNS)
    t = pq.read_table(path, columns=list(RESPONSE_COLUMNS),
                      read_dictionary=["subject_id", "item_id", "interactors"])
    y = t.column("response").to_numpy(zero_copy_only=False).astype(float)
    sid, iid, inter = (t.column(c).combine_chunks() for c in ("subject_id", "item_id", "interactors"))
    del t
    if sid.null_count or iid.null_count:
        raise PoolError(f"{b}: null subject_id or item_id in responses")
    R = _Resp()
    R.b, R._first, R._ifirst = b, None, None
    R.rows, R.nonnull = int(len(y)), int(np.sum(~np.isnan(y)))
    R.pos = np.flatnonzero((y == 0.0) | (y == 1.0))
    R.n01 = int(len(R.pos))
    R.y = y[R.pos].astype(np.int8)
    del y
    R.sids = [str(x) for x in sid.dictionary.to_pylist()]
    R.iids = [str(x) for x in iid.dictionary.to_pylist()]
    R.s = sid.indices.to_numpy(zero_copy_only=False)[R.pos].astype(np.int32)
    R.i = iid.indices.to_numpy(zero_copy_only=False)[R.pos].astype(np.int32)
    R.inames = [_text(x) for x in inter.dictionary.to_pylist()] + [""]
    R.ic = pc.fill_null(inter.indices, len(R.inames) - 1).to_numpy(
        zero_copy_only=False)[R.pos].astype(np.int32)
    return R


_RESP: OrderedDict = OrderedDict()


def _responses(b, root=None) -> _Resp:
    """_read_responses of a verified file; the last RESP_CACHE kept in memory."""
    path = verify(b, "responses.parquet", root)
    key = (path, _VERIFIED[path][0])
    if key in _RESP:
        _RESP.move_to_end(key)
        return _RESP[key]
    R = _read_responses(path, b)
    _RESP[key] = R
    while len(_RESP) > RESP_CACHE:
        _RESP.popitem(last=False)
    return R


def _pair_rows(R: _Resp):
    """[(subject_id, distinct items, first-response successes)] of the eligible
    pairs, in order of their first 0/1 response."""
    f = R.first()
    fs = R.s[f]
    n = np.bincount(fs, minlength=len(R.sids))
    k = np.bincount(fs, weights=R.y[f].astype(float), minlength=len(R.sids))
    _, at = np.unique(fs, return_index=True)
    order = fs[np.sort(at)]
    return [(R.sids[c], int(n[c]), int(round(k[c]))) for c in order.tolist() if n[c] >= MIN_ITEMS]


def _table_key(b, root=None) -> dict:
    return {"revision": REVISION, "benchmark": b, "min_items": MIN_ITEMS,
            "responses": file_sha256(b, "responses.parquet", root),
            "code": _source_sha(("_Resp", "_read_responses", "_pair_rows"))}


def pair_table(b, root=None, cache=True) -> dict:
    """The benchmark's response summary and eligible pairs, uncapped: rows,
    non-null and 0/1 responses, the 0/1 share, and per pair (subject_id,
    distinct items n, first-response successes k, corrected logit
    log((k + 0.5) / (n - k + 0.5))). Cached under FITS_DIR/tables/ by the
    responses file's sha256 and this code's digest; ids and counts only."""
    key = _table_key(b, root)
    path = os.path.join(FITS_DIR, "tables", f"{b}.json")
    if cache and os.path.exists(path):
        with open(path) as f:
            got = json.load(f)
        if got.get("key") == key:
            return _with_logits(got)
    R = _responses(b, root)
    out = {"key": key, "benchmark": b, "rows": R.rows, "nonnull": R.nonnull, "n01": R.n01,
           "share01": (R.n01 / R.nonnull) if R.nonnull else None,
           "pairs": [list(p) for p in _pair_rows(R)]}
    if cache:
        _write(path, _json_bytes(out))
    return _with_logits(out)


def _with_logits(t):
    t = dict(t)
    t["pairs"] = [(s, n, k, math.log((k + 0.5) / (n - k + 0.5))) for s, n, k in t["pairs"]]
    t["level_cc"] = float(np.mean([p[3] for p in t["pairs"]])) if t["pairs"] else None
    return t


def _table_task(task):
    b, root, fits_dir = task
    global FITS_DIR
    FITS_DIR = fits_dir
    t = pair_table(b, root)
    return b, {k: t[k] for k in ("rows", "nonnull", "n01", "share01")} | {"pairs": len(t["pairs"])}


def pair_tables(names, jobs=1, root=None) -> dict:
    """pair_table for every name, in up to `jobs` processes (caching each)."""
    todo = [b for b in sorted(names)]
    if jobs > 1 and len(todo) > 1:
        _pin_blas()
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(_table_task, [(b, _root(root), FITS_DIR) for b in todo]):
                pass
    return {b: pair_table(b, root) for b in todo}


def select(root=None, jobs=1, flagged=FLAGGED):
    """(L, U, report) by the rule of PE 3.3 (SELECTION_RULE), every name sorted.

    report holds, per catalogue entry, the reason it is in or out, its kind,
    direction, 0/1 share and eligible pairs, and the flagged set the
    catalogue's directions give (which the fits stage requires to equal
    FLAGGED)."""
    m = manifest(root)
    cand, why = [], {}
    for b in sorted(m["benchmarks"]):
        e = m["benchmarks"][b]
        if e.get("status") != "ok" or e.get("release") != "public":
            why[b] = {"in": False, "reason": f"status {e.get('status')}, release {e.get('release')}"}
        elif e.get("granularity") != "item":
            why[b] = {"in": False, "reason": f"granularity {e.get('granularity')}"}
        elif e.get("response_type") not in KINDS:
            why[b] = {"in": False, "reason": f"response_type {e.get('response_type')}"}
        else:
            cand.append(b)
    tables = pair_tables(cand, jobs, root)
    L, directions = [], {}
    for b in cand:
        cat, t = catalogue(b, root), tables[b]
        directions[b] = cat["direction"]
        info = {"kind": cat["response_type"], "direction": cat["direction"],
                "share01": t["share01"], "eligible_pairs": len(t["pairs"])}
        if cat["response_type"] == "mixed" and (t["share01"] is None or t["share01"] < SHARE01_MIN):
            why[b] = {"in": False, "reason": f"mixed, 0/1 share {t['share01']}", **info}
        elif not t["pairs"]:
            why[b] = {"in": False, "reason": "no eligible pair", **info}
        else:
            why[b] = {"in": True, "reason": "", **info}
            L.append(b)
    flagged = tuple(sorted(flagged))
    U = [b for b in L if b not in flagged]
    derived = sorted(b for b in L if directions.get(b) in FLAG_DIRECTIONS)
    rep = {"rule": SELECTION_RULE, "benchmarks": why, "flagged_by_direction": derived,
           "flagged": list(flagged), "flagged_match": derived == list(flagged),
           "L": len(L), "U": len(U)}
    return L, U, rep


def evaluation_set(L) -> list:
    return [b for b in sorted(L) if b not in LEGACY]


def levels(names, root=None) -> dict:
    """{benchmark: level_cc}, the mean corrected logit of the eligible pairs'
    first responses (PE 5.1's unit levels), from the verified pool files."""
    return {b: pair_table(b, root)["level_cc"] for b in sorted(names)}


def check_csvs(names, root=None, bench_csv=None, pairs_csv=None, tol=CSV_TOL) -> dict:
    """N3: the levels and pair table derived here against the planning CSVs
    (benchmarks.csv, pairs.csv of data/p2_work/pool_design/): the same
    eligible pairs with the same n and k, levels and logits within `tol`, the
    same 0/1 shares and eligible-pair counts. Raises PoolError on a
    difference; returns the largest differences."""
    import csv
    bench_csv = bench_csv or os.path.join(PLANNING, "benchmarks.csv")
    pairs_csv = pairs_csv or os.path.join(PLANNING, "pairs.csv")
    B = {r["benchmark"]: r for r in csv.DictReader(open(bench_csv))}
    P = defaultdict(dict)
    for r in csv.DictReader(open(pairs_csv)):
        P[r["benchmark"]][r["subject_id"]] = r
    worst = {"level": 0.0, "logit": 0.0, "share01": 0.0}
    problems = []
    for b in sorted(names):
        t = pair_table(b, root)
        if b not in B:
            problems.append(f"{b}: not in benchmarks.csv")
            continue
        r = B[b]
        if int(float(r["eligible_pairs"] or 0)) != len(t["pairs"]):
            problems.append(f"{b}: {len(t['pairs'])} pairs, csv {r['eligible_pairs']}")
        if r["share01"] and t["share01"] is not None:
            worst["share01"] = max(worst["share01"], abs(float(r["share01"]) - t["share01"]))
        if t["pairs"]:
            worst["level"] = max(worst["level"], abs(float(r["level_cc"]) - t["level_cc"]))
        mine = {s: (n, k, lg) for s, n, k, lg in t["pairs"]}
        if set(mine) != set(P[b]):
            problems.append(f"{b}: pair sets differ ({len(set(mine) ^ set(P[b]))})")
            continue
        for s, (n, k, lg) in mine.items():
            q = P[b][s]
            if int(q["n_items"]) != n or int(q["k_first"]) != k:
                problems.append(f"{b}/{s}: n, k {n}, {k}; csv {q['n_items']}, {q['k_first']}")
            worst["logit"] = max(worst["logit"], abs(float(q["logit_cc"]) - lg))
    if max(worst.values()) > tol:
        problems.append(f"largest differences {worst} above {tol}")
    if problems:
        raise PoolError("pool tables disagree with the planning CSVs:\n  " +
                        "\n  ".join(problems[:20]))
    return {"benchmarks": len(names), "pairs": sum(len(P[b]) for b in names),
            "max_abs_diff": worst, "tol": tol}


def check_selection_csv(L, bench_csv=None) -> dict:
    """L by the planning rule on benchmarks.csv (items_present.in_L) against L here."""
    import csv
    bench_csv = bench_csv or os.path.join(PLANNING, "benchmarks.csv")
    rows = list(csv.DictReader(open(bench_csv)))

    def in_l(r):
        if r["release"] != "public" or r["granularity"] != "item":
            return False
        if float(r["eligible_pairs"] or 0) < 1:
            return False
        return r["response_type"] == "binary" or (
            r["response_type"] == "mixed" and float(r["share01"] or 0) >= SHARE01_MIN)
    planned = sorted(r["benchmark"] for r in rows if in_l(r))
    if planned != sorted(L):
        raise PoolError(f"L differs from benchmarks.csv's: {sorted(set(planned) ^ set(L))}")
    return {"L": len(L), "equal": True}


# --- source families (PE 3.5, 3.6) -----------------------------------------------------

def src_key(url):
    """host (lower-cased, without www.) plus the first two path segments."""
    if not url or not str(url).strip():
        return None
    u = urlsplit(str(url).strip())
    host = u.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    segs = [s for s in u.path.split("/") if s][:2]
    return "src:" + "/".join([host] + segs)


def paper_key(url):
    """arXiv ids without version; OpenReview forum ids; DOIs; else host plus path."""
    if not url or not str(url).strip():
        return None
    s = str(url).strip()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", s, re.I)
    if m:
        return "paper:arxiv:" + m.group(1)
    u = urlsplit(s)
    host = u.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    if host == "openreview.net":
        q = parse_qs(u.query).get("id")
        if q:
            return "paper:openreview:" + q[0]
    if host in ("doi.org", "dx.doi.org"):
        return "paper:doi:" + u.path.strip("/").lower()
    return "paper:" + host + "/" + u.path.strip("/").lower()


def families(root=None) -> dict:
    """Source families over every catalogue entry of the manifest (FAMILY_RULE):
    family_of (benchmark -> its family's alphabetically first member), keys
    (each entry's joining keys), multi (families of two or more members) and
    joining_keys (the keys two or more members of a family share)."""
    m = manifest(root)
    names = sorted(b for b, e in m["benchmarks"].items()
                   if "benchmarks.parquet" in e.get("files", {}))
    keys = {}
    for b in names:
        row = catalogue(b, root)
        keys[b] = sorted(k for k in (src_key(row["source_url"]), paper_key(row["paper_url"])) if k)
    parent = {b: b for b in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for b in names:
        for k in keys[b]:
            if k in owner:
                ra, rb = find(owner[k]), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                owner[k] = b
    comp = defaultdict(list)
    for b in names:
        comp[find(b)].append(b)
    fam = {b: min(comp[find(b)]) for b in names}
    multi = {min(v): sorted(v) for v in comp.values() if len(v) > 1}
    shared = {}
    for f, members in multi.items():
        cnt = defaultdict(int)
        for b in members:
            for k in keys[b]:
                cnt[k] += 1
        shared[f] = sorted(k for k, c in cnt.items() if c > 1)
    return {"rule": FAMILY_RULE, "benchmarks": len(names), "families": len(comp),
            "multi_member_families": len(multi), "family_of": fam, "keys": keys,
            "multi": {f: multi[f] for f in sorted(multi)},
            "joining_keys": {f: shared[f] for f in sorted(shared)}}


def excl_for(benchmarks, family_of, L) -> tuple:
    """Every benchmark of L in the family of a benchmark of `benchmarks` (PE 3.6)."""
    fams = {family_of[b] for b in benchmarks}
    return tuple(sorted(b for b in L if family_of[b] in fams))


def check_families(fam, planned=None) -> dict:
    """The families against the planning twin, a4final/families.json (PE 3.5)."""
    planned = planned or os.path.join(PLANNING, "a4final", "families.json")
    with open(planned) as f:
        p = json.load(f)
    diffs = [k for k in ("family_of", "multi", "joining_keys") if p[k] != fam[k]]
    if diffs or p["families"] != fam["families"]:
        raise PoolError(f"families differ from {planned} in {diffs or ['families']}")
    return {"equal": True, "families": fam["families"], "multi": fam["multi_member_families"]}


# --- subjects, items, pairs (PE 3.2, 3.4) -----------------------------------------------

def subjects_of(b, root=None) -> dict:
    """{subject_id: the eight visible fields}; display_name is never read."""
    import pyarrow.parquet as pq
    path = verify(b, "subjects.parquet", root)
    cols = ["subject_id"] + list(SUBJECT_FIELDS)
    _require(path, cols)
    out = {}
    for r in pq.read_table(path, columns=cols).to_pylist():
        sid = _text(r["subject_id"])
        if sid not in out:
            out[sid] = {f: _text(r[f]) for f in SUBJECT_FIELDS}
    return out


def _subject(subjects, sid) -> dict:
    d = dict(subjects.get(sid) or {f: "" for f in SUBJECT_FIELDS})
    d["_sid"] = sid
    return d


def items_of(b, item_ids, root=None, content=True) -> dict:
    """{item_id: (content, item_features)} for these items of items.parquet,
    '' for nulls (PE 3.2, N8). It reads exactly ITEM_COLUMNS, in batches of at
    most ITEM_BATCH rows through a READ_BUFFER-byte stream with pre_buffer
    off, keeps the rows of the requested ids and drops each batch after use,
    so no column chunk is held whole; never verifier or grading_criterion,
    never a whole-table read. content=False keeps '' for the content (the fits:
    benchmark_fit reads no text). Raises PoolError for an item the table
    lacks or a missing column."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = sorted({str(i) for i in item_ids})
    if not want:
        return {}
    path = verify(b, "items.parquet", root)
    _require(path, ITEM_COLUMNS)
    pf = pq.ParquetFile(path, buffer_size=READ_BUFFER, pre_buffer=False)
    wanted = pa.array(want, pa.string())
    out = {}
    try:
        for batch in pf.iter_batches(batch_size=ITEM_BATCH, columns=list(ITEM_COLUMNS),
                                     use_threads=False):
            hit = pc.is_in(batch.column(0), value_set=wanted)
            if pc.any(hit).as_py():
                sub = batch.filter(hit).to_pydict()
                for j, iid in enumerate(sub["item_id"]):
                    iid = str(iid)
                    if iid not in out:
                        out[iid] = (_text(sub["content"][j]) if content else "",
                                    _text(sub["item_features"][j]))
                del sub
            del batch, hit
            if len(out) == len(want):
                break
    finally:
        pf.close()
    missing = [i for i in want if i not in out]
    if missing:
        raise PoolError(f"{b}: {len(missing)} item(s) not in items.parquet, e.g. {missing[0]}")
    return out


def _item_dicts(b, R: _Resp, codes, root=None, content=True) -> dict:
    """{item code: the Mode A item dict}: content and features from items_of,
    interactors from the item's first 0/1 response row, benchmark-wide."""
    codes = sorted(set(int(c) for c in codes))
    first = R.item_first_row()
    got = items_of(b, [R.iids[c] for c in codes], root, content)
    out = {}
    for c in codes:
        iid = R.iids[c]
        text, feats = got[iid]
        out[c] = {"item_content": text, "item_features": feats,
                  "interactors": R.inames[R.ic[first[c]]], "benchmark_id": b, "_key": iid}
    return out


def cap_order(b, subjects) -> list:
    """The pair cap's order (PE 3.4): round robin over canonical names, ascending
    (j, stable_hash('pool-name', c)), c the pair's canon_name (its subject_id
    when it has none) and j its rank among the benchmark's pairs of the same c
    by stable_hash('pool-pair', b, subject_id). `subjects` is
    [(subject_id, subject dict)]; returns the subject_ids in cap order."""
    by = defaultdict(list)
    for sid, s in subjects:
        c = PR.canon_name(s) or sid
        by[c].append(sid)
    keyed = []
    for c in sorted(by):
        ranked = sorted(by[c], key=lambda x: (stable_hash("pool-pair", b, x), x))
        h = stable_hash("pool-name", c)
        for j, sid in enumerate(ranked):
            keyed.append(((j, h, c, sid), sid))
    return [sid for _, sid in sorted(keyed)]


def item_rank(b, iids) -> dict:
    """{item_id: rank} in the benchmark-wide order of the per-pair item cap,
    ascending stable_hash('pool-item', b, item_id)."""
    order = sorted(iids, key=lambda x: (stable_hash("pool-item", b, x), x))
    return {x: r for r, x in enumerate(order)}


def _fit_cells(b, k=K, m_p=M_P, root=None):
    """What load_fit_pairs keeps, before any item is read: (R, subjects, the
    kept subject codes in order of first 0/1 response, {code: indices into
    the first cells}, the first cells' item codes and labels)."""
    R = _responses(b, root)
    fs, fi, fy, cells = R.cells_by_subject()
    _, at = np.unique(fs, return_index=True)
    order = [c for c in fs[np.sort(at)].tolist() if len(cells[c]) >= MIN_ITEMS]
    subjects = subjects_of(b, root)
    if not order:
        return R, subjects, [], {}, fi, fy
    chosen = set(cap_order(b, [(R.sids[c], _subject(subjects, R.sids[c])) for c in order])[:k])
    order = [c for c in order if R.sids[c] in chosen]
    present = np.unique(np.concatenate([fi[cells[c]] for c in order]))
    rank = item_rank(b, [R.iids[c] for c in present.tolist()])
    r_of = np.full(len(R.iids), -1, np.int64)
    for c in present.tolist():
        r_of[c] = rank[R.iids[c]]
    keep = {}
    for c in order:
        idx = cells[c]
        if len(idx) > m_p:
            r = r_of[fi[idx]]
            idx = idx[r <= np.partition(r, m_p - 1)[m_p - 1]]
        keep[c] = idx
    return R, subjects, order, keep, fi, fy


def _fit_pairs(b, cells, root=None, content=False) -> list:
    R, subjects, order, keep, fi, fy = cells
    if not order:
        return []
    items = _item_dicts(b, R, np.concatenate([fi[keep[c]] for c in order]), root, content)
    pairs = []
    for c in order:
        sid = R.sids[c]
        rs = [Response(item_key=R.iids[int(fi[j])], item=items[int(fi[j])], label=int(fy[j]))
              for j in keep[c].tolist()]
        pairs.append(Pair(subject=_subject(subjects, sid), subject_id=sid, benchmark_id=b,
                          responses=rs))
    return pairs


def load_fit_pairs(b, k=K, m_p=M_P, root=None, content=True) -> list:
    """The pairs one benchmark's fit sees (PE 3.4): each eligible pair's first
    0/1 response per item in file order; at most m_p items a pair, its first in
    item_rank's order; at most k pairs, the first in cap_order; pairs in order
    of their first 0/1 response; Mode A items. paiec.prior.benchmark_fit reads
    only an item's features and interactors (hier.item_groups without the
    text prefix), so fit_one asks for content=False and the fit is the same
    (tests/test_pool.py checks it)."""
    return _fit_pairs(b, _fit_cells(b, k, m_p, root), root, content)


def pair_counts(b, root=None) -> list:
    """[(subject_id, n, k, logit)] of every eligible pair (N3, PE 5.1): n its
    distinct items with a 0/1 response, k those whose first 0/1 response in
    file order is 1, logit log((k + 0.5) / (n - k + 0.5)); from the verified
    responses through pair_table."""
    return pair_table(b, root)["pairs"]


def pair_items(b, root=None) -> dict:
    """{subject_id: its distinct item ids in order of first 0/1 response} for
    every eligible pair, uncapped: what a run sampler cuts before any content
    is read."""
    R = _responses(b, root)
    _, fi, _, cells = R.cells_by_subject()
    code = {s: c for c, s in enumerate(R.sids)}
    return {sid: [R.iids[j] for j in fi[cells[code[sid]]].tolist()] for sid, _, _ in _pair_rows(R)}


# --- the evaluation loader (EVAL_LOADER: the only definitions a post-lock fix may
# change, PE 9 item 11; nothing outside it may use them but main) -----------------------

INDEX_VERSION = 1


def _eval_index_path(b) -> str:
    return os.path.join(EVAL_DIR, "index", f"{b}.npz")


def eval_index(b, root=None, build=True) -> dict:
    """The evaluation loader's index of one benchmark, ids and offsets only,
    cached in EVAL_DIR/index/<b>.npz under the responses file's sha256: the
    eligible pairs' subject ids and distinct-item counts, the runs (first
    file row, length) of each pair's consecutive 0/1 rows, and every item's
    id with the file row of its first 0/1 response (benchmark-wide, for its
    interactors). Built from a full read, so build the indexes before scoring
    (python -m paiec.pool --index)."""
    sha = file_sha256(b, "responses.parquet", root)
    path = _eval_index_path(b)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            if str(z["sha256"]) == sha and int(z["version"]) == INDEX_VERSION:
                return {k: z[k] for k in z.files}
    if not build:
        raise PoolError(f"{b}: no eval index at {path}")
    R = _read_responses(verify(b, "responses.parquet", root), b)
    eligible = _pair_rows(R)
    code = {s: c for c, s in enumerate(R.sids)}
    order = np.argsort(R.s, kind="stable")
    counts = np.bincount(R.s, minlength=len(R.sids))
    ends = np.cumsum(counts)
    starts, lens, sub = [], [], [0]
    for sid, _, _ in eligible:
        c = code[sid]
        p = R.pos[order[ends[c] - counts[c]:ends[c]]]
        brk = np.flatnonzero(np.diff(p) != 1) + 1
        st = np.concatenate([[0], brk])
        en = np.concatenate([brk, [len(p)]])
        starts.append(p[st])
        lens.append(en - st)
        sub.append(sub[-1] + len(st))
    _, at = np.unique(R.i, return_index=True)
    ix = {"version": np.array(INDEX_VERSION), "sha256": np.array(sha),
          "sids": np.array([s for s, _, _ in eligible] or [""], dtype=np.str_)[:len(eligible)],
          "n_items": np.array([n for _, n, _ in eligible], np.int64),
          "run_start": np.concatenate(starts).astype(np.int64) if starts else np.zeros(0, np.int64),
          "run_len": np.concatenate(lens).astype(np.int64) if lens else np.zeros(0, np.int64),
          "sub_runs": np.array(sub, np.int64),
          "iids": np.array([R.iids[c] for c in R.i[at].tolist()] or [""], dtype=np.str_)[:len(at)],
          "item_first_pos": R.pos[at].astype(np.int64)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **ix)
    os.replace(tmp, path)
    return ix


def _eval_rows(pf, bounds, rows, cols) -> dict:
    """{column: an Arrow array of its values at these file rows (sorted,
    unique)}, reading only the row groups that hold them."""
    import pyarrow as pa
    rows = np.asarray(rows, np.int64)
    if not len(rows):
        return {c: pa.array([], pa.string()) for c in cols}
    g = np.searchsorted(bounds, rows, side="right") - 1
    rgs = np.unique(g)
    t = pf.read_row_groups(rgs.tolist(), columns=list(cols), use_threads=False)
    size = bounds[rgs + 1] - bounds[rgs]
    base = np.zeros(int(rgs.max()) + 1, np.int64)
    base[rgs] = np.cumsum(size) - size
    take = pa.array(base[g] + rows - bounds[g])
    return {c: t.column(c).take(take).combine_chunks() for c in cols}


def load_eval_pairs(b, subject_ids, keep=None, root=None) -> list:
    """The drawn pairs of an evaluation run (PE 5.2): every 0/1 response in file
    order, repeats included, uncapped; `keep` ({subject_id: item ids}) limits a
    pair to those items. Mode A items: content and features through items_of,
    interactors from the item's first 0/1 response row, benchmark-wide. Reads
    only the row groups that hold the pairs' rows (eval_index). Raises
    PoolError for a pair that is not eligible, an index that disagrees with
    the file, or more than MAX_CONTENT_ITEMS items to read (N8: pass keep)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    ix = eval_index(b, root)
    path = verify(b, "responses.parquet", root)
    at = {s: j for j, s in enumerate(ix["sids"].tolist())}
    pf = pq.ParquetFile(path)
    md = pf.metadata
    bounds = np.cumsum([0] + [md.row_group(g).num_rows for g in range(md.num_row_groups)])
    got = {}
    for sid in subject_ids:
        if sid not in at:
            raise PoolError(f"{b}/{sid}: not an eligible pair")
        j = at[sid]
        a, z = int(ix["sub_runs"][j]), int(ix["sub_runs"][j + 1])
        rows = np.concatenate([np.arange(st, st + ln) for st, ln in
                               zip(ix["run_start"][a:z].tolist(), ix["run_len"][a:z].tolist())])
        v = _eval_rows(pf, bounds, rows, ("subject_id", "item_id", "response"))
        ok = pc.and_(pc.equal(v["subject_id"], sid),
                     pc.or_(pc.equal(v["response"], 0.0), pc.equal(v["response"], 1.0)))
        if not pc.all(ok).as_py() or \
                pc.count_distinct(v["item_id"]).as_py() != int(ix["n_items"][j]):
            raise PoolError(f"{b}/{sid}: the eval index disagrees with the responses file")
        iid, y = v["item_id"], v["response"]
        if keep is not None and sid in keep:
            hit = pc.is_in(iid, value_set=pa.array(sorted({str(x) for x in keep[sid]}), pa.string()))
            iid, y = iid.filter(hit), y.filter(hit)
        got[sid] = ([str(x) for x in iid.to_pylist()], [int(x) for x in y.to_pylist()])
    need = sorted({x for iid, _ in got.values() for x in iid})
    if len(need) > MAX_CONTENT_ITEMS:
        raise PoolError(f"{b}: {len(need)} items' content asked for, more than "
                        f"{MAX_CONTENT_ITEMS}; pass keep")
    first = dict(zip(ix["iids"].tolist(), ix["item_first_pos"].tolist()))
    v = _eval_rows(pf, bounds, sorted({first[x] for x in need}), ("item_id", "interactors"))
    inter = {str(i): _text(x) for i, x in zip(v["item_id"].to_pylist(), v["interactors"].to_pylist())}
    if set(inter) != set(need):
        raise PoolError(f"{b}: the eval index's first rows disagree with the responses file")
    texts = items_of(b, need, root)
    items = {x: {"item_content": texts[x][0], "item_features": texts[x][1],
                 "interactors": inter[x], "benchmark_id": b, "_key": x} for x in need}
    subjects = subjects_of(b, root)
    out = []
    for sid in subject_ids:
        iid, y = got[sid]
        rs = [Response(item_key=x, item=items[x], label=lab) for x, lab in zip(iid, y)]
        out.append(Pair(subject=_subject(subjects, sid), subject_id=sid, benchmark_id=b,
                        responses=rs))
    return out


def build_eval_indexes(names, jobs=1, root=None) -> dict:
    """eval_index for every name, in up to `jobs` fresh processes; their sha256."""
    todo = sorted(names)
    if jobs > 1 and len(todo) > 1:
        _pin_blas()
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(_eval_index_task, [(b, _root(root), EVAL_DIR) for b in todo]):
                pass
    out = {}
    for b in todo:
        eval_index(b, root)
        out[b] = _sha256_file(_eval_index_path(b))
    return out


def _eval_index_task(task):
    b, root, eval_dir = task
    global EVAL_DIR
    EVAL_DIR = eval_dir
    eval_index(b, root)
    return b


#: the evaluation loader: the definitions PE 9 item 11 lets a post-lock fix change (this
#: list itself is frozen, so the scope cannot widen after the lock)
EVAL_LOADER = ("INDEX_VERSION", "_eval_index_path", "eval_index", "_eval_rows",
               "load_eval_pairs", "build_eval_indexes", "_eval_index_task")


def frozen_ast(source) -> str:
    """ast.dump of pool.py's source without the EVAL_LOADER definitions: what a
    post-lock fix must leave unchanged (PE 9 item 11)."""
    import ast
    tree = ast.parse(source)

    def named(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None
    tree.body = [n for n in tree.body if named(n) not in EVAL_LOADER]
    return ast.dump(tree, include_attributes=False)


def subject_sets(names, root=None, years=DATE_SHIFT):
    """(by_real, by_shifted): {benchmark: [subject dict of each eligible pair]}
    over `names`, uncapped, at real dates and with release and access dates
    moved by `years` (paiec.testlike.shift_date; undated subjects do not move)."""
    from paiec.testlike import shift_date
    real, ds = {}, {}
    for b in sorted(names):
        subjects = subjects_of(b, root)
        real[b] = [_subject(subjects, s) for s, _, _, _ in pair_table(b, root)["pairs"]]
        ds[b] = []
        for s in real[b]:
            t = dict(s)
            for f in ("release_date", "access_date"):
                t[f] = shift_date(t[f], years)
            ds[b].append(t)
    return real, ds


# --- fits (PE 4.1 step 1, 4.3; cached, PE 9 item 3) -------------------------------------

#: what a cached fit holds: benchmark_fit's numbers and the subject dicts (eight
#: visible fields and _sid), no item text
FIT_KEEP = ("benchmark", "n_pairs", "n_items", "n_cells", "mu", "mu_var", "t", "t_var",
            "subjects", "s2_t", "s2e", "s2_g", "keys", "item_var", "share", "iterations",
            "converged", "newton_converged")


def _pin_blas():
    for v in BLAS_VARS:
        os.environ.setdefault(v, "1")


def _fit_key(b, s2e, k, m_p, root=None) -> dict:
    return {"revision": REVISION, "benchmark": b, "K": k, "M_P": m_p,
            "s2e": None if s2e is None else repr(float(s2e)),
            "files": {t: file_sha256(b, t, root)
                      for t in ("responses.parquet", "subjects.parquet", "items.parquet")},
            "loader": source_digest("loader"),
            "prior_py": _sha256_file(os.path.join(ROOT, "paiec", "prior.py")),
            "hier_py": _sha256_file(os.path.join(ROOT, "paiec", "hier.py"))}


def _fit_path(fits_dir, key) -> str:
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
    return os.path.join(fits_dir, key["benchmark"], f"{h}.json")


def fit_one(b, s2e=None, k=K, m_p=M_P, root=None) -> dict:
    """benchmark_fit on load_fit_pairs' pairs (content not read: the fit reads no
    text), with the CPU time of the response read, of the item read and of the
    fit, and the process's peak resident set (N8)."""
    c0, w0 = time.process_time(), time.perf_counter()
    cells = _fit_cells(b, k, m_p, root)
    c1 = time.process_time()
    pairs = _fit_pairs(b, cells, root, content=False)
    del cells
    c2, w2 = time.process_time(), time.perf_counter()
    if not pairs:
        raise PoolError(f"{b}: no eligible pair")
    if len(pairs) < 2 and s2e is None:
        raise PoolError(f"{b}: one pair, so the item variance must be given")
    f = PR.benchmark_fit(pairs, s2e=s2e)
    c3, w3 = time.process_time(), time.perf_counter()
    out = {k_: f[k_] for k_ in FIT_KEEP}
    out.update(s2e_given=None if s2e is None else float(s2e), response_cpu=c1 - c0,
               item_cpu=c2 - c1, load_cpu=c2 - c0, load_wall=w2 - w0, fit_cpu=c3 - c2,
               fit_wall=w3 - w2, maxrss_mb=_maxrss_mb(), pid=os.getpid())
    return out


def _fit_task(task):
    b, s2e, k, m_p, root, fits_dir = task
    global FITS_DIR
    FITS_DIR = fits_dir
    return b, s2e, fit_one(b, s2e, k, m_p, root)


def _cached(fits_dir, key):
    path = _fit_path(fits_dir, key)
    if os.path.exists(path):
        with open(path) as f:
            got = json.load(f)
        if got.get("key") == key:
            return got["fit"]
    return None


def _run_fits(tasks, jobs, root, fits_dir, cache, log, fresh=True):
    """{(b, s2e): fit} for tasks [(b, s2e)], cached ones read back, the rest in
    up to `jobs` fresh processes (one task each, so each peak RSS is its own)."""
    out, todo = {}, []
    for b, s2e in tasks:
        key = _fit_key(b, s2e, K, M_P, root)
        got = _cached(fits_dir, key) if cache else None
        if got is not None:
            out[(b, s2e)] = got
        else:
            todo.append((b, s2e, key))
    if not todo:
        return out
    _pin_blas()
    args = [(b, s2e, K, M_P, _root(root), fits_dir) for b, s2e, _ in todo]
    keys = {(b, s2e): key for b, s2e, key in todo}
    if fresh:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(max(1, jobs), maxtasksperchild=1) as pool:
            it = pool.imap_unordered(_fit_task, args)
            for b, s2e, f in it:
                out[(b, s2e)] = _store(fits_dir, keys[(b, s2e)], f)
                log(f"  {b}{'' if s2e is None else ' (fixed T)'}: {f['n_pairs']} pairs, "
                    f"{f['n_items']} items, {f['iterations']} it, {f['fit_cpu']:.1f} s, "
                    f"{f['maxrss_mb']:.0f} MB")
    else:
        for a in args:
            b, s2e, f = _fit_task(a)
            out[(b, s2e)] = _store(fits_dir, keys[(b, s2e)], f)
    return out


def _store(fits_dir, key, fit):
    """Write a fit to the cache and return it as the cache gives it back, so a
    fresh and a cached run hand the bundle the same numbers."""
    data = _json_bytes({"key": key, "fit": fit})
    _write(_fit_path(fits_dir, key), data)
    return json.loads(data)["fit"]


def degenerate(free) -> list:
    """Amendment 4a: the benchmarks whose free fit (two or more pairs after the
    caps) has an item variance above DEGENERATE_ITEM_VAR, sorted. Each is left
    out of every pooled hyperparameter (the item variance T and T_full, the
    group share, sigma_delta and sigma_attr, C2's level distribution, C1m's
    matching, and by amendment 4b the ridge's s2_res and so u); its pairs
    still serve the subject standings and the identity table, and it stays in
    L, U and E."""
    return sorted(b for b, f in free.items()
                  if f["n_pairs"] >= 2 and f["item_var"] > DEGENERATE_ITEM_VAR)


def fit_all(names, jobs=1, root=None, cache=True, log=print, fresh=True):
    """(free, fixed, T_full) for the benchmarks `names` (L): free-item-variance
    fits of every benchmark with two or more pairs after the caps; T_full, the
    median of their item variances, the degenerate fits (amendment 4a,
    degenerate()) left out; the fixed-T pass, benchmark_fit(ps,
    s2e=T_full), for every benchmark. Each fit runs in a fresh process (one BLAS
    thread, maxtasksperchild 1) unless fresh=False, so its peak resident set is
    its own (N8). Cached under FITS_DIR; FITS_DIR/index.json lists every entry
    used with its sha256."""
    names = sorted(names)
    tables = {b: pair_table(b, root) for b in names}
    multi = [b for b in names if min(K, len(tables[b]["pairs"])) >= 2]
    size = {b: sum(min(M_P, n) for _, n, _, _ in tables[b]["pairs"][:K]) for b in names}
    order = sorted(multi, key=lambda b: (-size[b], b))
    got = _run_fits([(b, None) for b in order], jobs, root, FITS_DIR, cache, log, fresh)
    free = {b: got[(b, None)] for b in multi}
    deg = set(degenerate(free))
    T_full = float(np.median([free[b]["item_var"] for b in multi if b not in deg]))
    order = sorted(names, key=lambda b: (-size[b], b))
    got = _run_fits([(b, T_full) for b in order], jobs, root, FITS_DIR, cache, log, fresh)
    fixed = {b: got[(b, T_full)] for b in names}
    index = {"revision": REVISION, "K": K, "M_P": M_P, "T_full": repr(T_full),
             "loader": source_digest("loader"), "entries": {}}
    for kind, s2e, bs in (("free", None, multi), ("fixed", T_full, names)):
        for b in bs:
            path = _fit_path(FITS_DIR, _fit_key(b, s2e, K, M_P, root))
            index["entries"][f"{b}/{kind}"] = {"file": os.path.relpath(path, FITS_DIR),
                                               "sha256": _sha256_file(path)}
    _write(os.path.join(FITS_DIR, "index.json"), _json_bytes(index))
    return free, fixed, T_full


def load_fits(root=None):
    """(free, fixed, T_full) from FITS_DIR/index.json, every entry's sha256
    checked; raises PoolError when the index or an entry is missing or changed."""
    path = os.path.join(FITS_DIR, "index.json")
    if not os.path.exists(path):
        raise PoolError(f"{path}: no fits; run python -m paiec.pool --fit")
    with open(path) as f:
        index = json.load(f)
    if index.get("loader") != source_digest("loader"):
        raise PoolError("the fits were made by another loader; refit")
    free, fixed = {}, {}
    for name, e in sorted(index["entries"].items()):
        b, kind = name.rsplit("/", 1)
        p = os.path.join(FITS_DIR, e["file"])
        if not os.path.exists(p) or _sha256_file(p) != e["sha256"]:
            raise PoolError(f"{p}: missing or changed since the index was written")
        with open(p) as f:
            (free if kind == "free" else fixed)[b] = json.load(f)["fit"]
    return free, fixed, float(index["T_full"])


def fits_index_sha256() -> str:
    return _sha256_file(os.path.join(FITS_DIR, "index.json"))


# --- the bundle arithmetic (PE 4.1-4.3; ARITHMETIC) ------------------------------------

def standings_rows(fits, names) -> list:
    """paiec.prior.standings' rows from cached fits: per benchmark of `names`
    with two or more pairs, (benchmark, subject, centred standing, variance)."""
    rows = []
    for b in sorted(names):
        f = fits[b]
        if f["n_pairs"] < 2:
            continue
        t = np.array(f["t"])
        for s, z, v in zip(f["subjects"], t - t.mean(), f["t_var"]):
            rows.append((b, s, float(z), float(v)))
    return rows


def pooled_meta(rows, prior, deg) -> dict:
    """Amendment 4a: a copy of the subject prior's meta whose pooled variances
    (s2_res, u_mean, total_var), the ones fit_hyper's arithmetic turns into
    sigma_delta and sigma_attr, are those paiec.prior.from_standings gives on
    the standings rows without the degenerate benchmarks `deg`, as if they
    were absent (amendment 4a's "2.04 without them"; amendment 4b reads
    sigma_delta so, 2.040). tau2_res, s2d and the identity table are
    unchanged (from_standings on every row: their pairs still serve the
    standings and the identity table); pooled_prior puts these pooled
    variances, and the coefficient covariance at this s2_res, into the prior
    (amendment 4b). A copy of prior.meta unchanged when no row lies on a
    degenerate benchmark (or every row does, leaving nothing to pool)."""
    meta = dict(prior.meta)
    out = set(deg)
    sub = [r for r in rows if r[0] not in out]
    if len(sub) == len(rows) or not sub:
        return meta
    m = PR.from_standings(sub, ALPHA).meta
    meta.update(s2_res=m["s2_res"], u_mean=m["u_mean"], total_var=m["total_var"],
                pooled_rows=len(sub))
    return meta


def pooled_prior(rows, prior, meta):
    """(prior, report). Amendment 4b: the attribute ridge's residual variance s2_res
    is a pooled hyperparameter, so it, and through it every subject's attribute
    variance u = x' C x, is computed without the degenerate benchmarks. The
    returned prior is `prior` (from_standings on every row of `rows`) with its
    coefficient covariance C = max(s2_res, 1e-6) (X'X + alpha I)^-1 X'X
    (X'X + alpha I)^-1, from_standings' own formula on the same rows' design, at
    meta's pooled s2_res (pooled_meta) instead of the all-row one; and with the
    pooled variances of its meta (s2_res, u_mean, total_var) pooled_meta's, so
    that fit_hyper's arithmetic on the prior's own meta gives the bundle's
    sigma_delta and sigma_attr (read as if the degenerate benchmarks were
    absent). The coefficients, the spec, the identity table, s2d and every
    other meta entry (tau2_res among them: the identity components) are the
    all-row prior's, and meta keeps its keys. `prior` itself, and no report,
    when meta pools nothing (no row lies on a degenerate benchmark)."""
    if prior is None or meta is None or meta.get("pooled_rows") is None:
        return prior, None
    coef, spec, X, Ai = PR._ridge(rows, ALPHA)
    if not np.array_equal(coef, prior.coef) or spec.to_dict() != prior.spec.to_dict():
        raise PoolError("the attribute ridge on the standings rows is not the prior's")
    s2 = float(meta["s2_res"])
    cov = max(s2, 1e-6) * Ai @ X.T @ X @ Ai
    u = np.einsum("ij,jk,ik->i", X, cov, X)
    new = dict(prior.meta)
    new.update(s2_res=meta["s2_res"], u_mean=meta["u_mean"], total_var=meta["total_var"])
    out = PR.SubjectPrior(prior.coef, prior.spec, cov, prior.table, prior.s2d, new)
    rep = {"s2_res": s2, "s2_res_all_rows": prior.meta.get("s2_res"),
           "u_mean_rows": float(u.mean()), "u_mean_all_rows": prior.meta.get("u_mean"),
           "total_var_all_rows": prior.meta.get("total_var"), "rows": len(rows),
           "pooled_rows": meta["pooled_rows"]}
    return out, rep


def match_set(by_benchmark, free) -> dict:
    """Amendment 4a: the pairs C1m's matching reads (and the centre table's
    "matching set"): by_benchmark without the degenerate benchmarks."""
    out = set(degenerate(free))
    return {b: v for b, v in by_benchmark.items() if b not in out}


def hyper_arith(fits, by, prior, excl=(), single=None, deg=(), meta=None):
    """(Hyper, report): paiec.prior.fit_hyper's arithmetic (paiec/prior.py
    :756-807 at a725df9, centre 'mean', nu_mu 0, its WIDEN and G_CAP) on fit
    dicts. fits[b] for every b of `by`, the included benchmarks: free fits for
    those with two or more pairs. The levels (mu0, sigma_mu) are fit_hyper's
    only when `single` is given, a function (b, T) -> the fit of a one-pair
    benchmark at item variance T; otherwise they are left to REFERENCE for the
    candidate to overwrite.

    Amendment 4a: the degenerate benchmarks `deg` (degenerate()) are left out
    of T and the group share, and `meta` (pooled_meta(): the prior's pooled
    variances without their rows) replaces prior.meta for sigma_delta and
    sigma_attr; sigma_theta keeps the prior's own tau2_res (the identity
    table). With deg and meta empty this is fit_hyper's arithmetic exactly."""
    from paiec.hier import Hyper
    held = sorted(set(excl))
    by = list(by)
    ref = dict(PR.REFERENCE)
    multi = {b: fits[b] for b in by if fits[b]["n_pairs"] >= 2}
    pooled = {b: f for b, f in multi.items() if b not in set(deg)}
    est, rep = {}, {"benchmarks": by, "excluded": held, "centre": "mean", "approximate": [],
                    "degenerate": sorted(set(deg) & set(multi))}
    if pooled:
        T = float(np.median([f["item_var"] for f in pooled.values()]))
        shares = [f["share"] for f in pooled.values() if f["share"] is not None]
        share = min(float(np.median(shares)), PR.G_CAP) if shares else \
            ref["sigma_g"] ** 2 / (ref["sigma_g"] ** 2 + ref["sigma_d"] ** 2)
        est["sigma_g"], est["sigma_d"] = math.sqrt(share * T), math.sqrt((1 - share) * T)
        rep.update(item_var=T, share=share, shares=shares)
    else:
        T = ref["sigma_d"] ** 2 + ref["sigma_g"] ** 2
    if single is not None:
        lf = dict(multi)
        for b in by:
            if b not in lf:
                lf[b] = single(b, T)
        attr = prior is not None and prior.has_attributes
        lev = {}
        for b in by:
            shift = float(np.mean(lf[b]["t"]))
            if attr:
                shift -= float(np.mean([prior.attribute(s)[0] for s in lf[b]["subjects"]]))
            lev[b] = lf[b]["mu"] + shift
        mus = np.array([lev[b] for b in by])
        n = len(mus)
        if n >= PR.MIN_LEVELS:
            est["mu0"] = float(np.mean(mus))
            est["sigma_mu"] = float(np.std(mus, ddof=1) * math.sqrt(1 + 1 / n)) * PR.WIDEN[0]
        rep["levels"], rep["levels_fitted"] = lev, {b: lf[b]["mu"] for b in by}
    own = prior.meta if prior is not None else {}
    meta = own if meta is None else meta
    th2 = ref["sigma_theta"] ** 2
    if own.get("tau2_res") is not None:
        th2 = float(np.clip(own["tau2_res"], ID_CLIP[0], ID_CLIP[1]))
        est["sigma_theta"] = math.sqrt(th2)
    if meta.get("s2_res") is not None:
        est["sigma_delta"] = math.sqrt(max(meta["s2_res"] - meta["u_mean"] - th2, 0.1)) * PR.WIDEN[1]
        est["sigma_attr"] = math.sqrt(max(meta["total_var"] - meta["s2_res"], 0.0))
    elif meta.get("total_var") is not None:
        est["sigma_delta"] = math.sqrt(max(meta["total_var"] - meta["u_mean"] - th2, 0.1)) * PR.WIDEN[1]
        rep["approximate"].append("sigma_delta")
    fallback = {k: v for k, v in ref.items() if k not in est}
    rep["estimated"], rep["fallback"] = sorted(est), sorted(fallback)
    rep["tau2_res"] = own.get("tau2_res")
    if meta is not own:
        rep["pooled"] = {k: meta.get(k) for k in ("s2_res", "u_mean", "total_var", "pooled_rows")}
    hyper = Hyper(**{**Hyper().to_dict(), **fallback, **est, "nu_mu": 0.0,
                     "excluded": held, "included": sorted(by)})
    return hyper, rep


def eb(levels, var):
    """(m, s2, iterations): the precision-weighted moment iteration of PE 4.3
    (pool_design/code/code_levels.py's eb): s2 from max(var(levels, ddof 1) -
    mean(var), EB_FLOOR), then at most EB_MAX_ITER times w = 1/(s2 + v),
    m = sum(w level)/sum(w), s2 = max(sum(w((level - m)^2 - v))/sum(w), EB_FLOOR),
    stopping when s2 moves by less than EB_TOL."""
    x = np.asarray(levels, float)
    v = np.asarray(var, float)
    s2 = max(float(x.var(ddof=1) - v.mean()), EB_FLOOR)
    m, it = float("nan"), 0
    for it in range(1, EB_MAX_ITER + 1):
        w = 1 / (s2 + v)
        m = float(np.sum(w * x) / w.sum())
        new = max(float(np.sum(w * ((x - m) ** 2 - v)) / w.sum()), EB_FLOOR)
        if abs(new - s2) < EB_TOL:
            break
        s2 = new
    return m, s2, it


def eb_level(level_fits, prior, names) -> dict:
    """C2's level (PE 4.3) over `names`: level_b = mu + mean(t) of the fixed-T
    fit, less the mean attribute score of its subjects under `prior`;
    v_b = mu_var; |level_b| > LEVEL_DROP dropped; eb() on the rest; mu0 = m,
    sigma_mu = sqrt(s2) sqrt(1 + 1/n). Also the size of the omitted Var(mean
    t) term, the mean of s2_t / n_pairs over the levels used."""
    attr = prior is not None and prior.has_attributes
    lv, vv, dropped, omit = {}, {}, [], []
    for b in sorted(names):
        g = level_fits[b]
        shift = float(np.mean(g["t"]))
        if attr:
            shift -= float(np.mean([prior.attribute(s)[0] for s in g["subjects"]]))
        level = g["mu"] + shift
        if not math.isfinite(level) or abs(level) > LEVEL_DROP:
            dropped.append(b)
            continue
        lv[b], vv[b] = level, g["mu_var"]
        omit.append((g["s2_t"] or 0.0) / g["n_pairs"])
    used = sorted(lv)
    if len(used) < PR.MIN_LEVELS:
        raise PoolError(f"{len(used)} levels left, fewer than {PR.MIN_LEVELS}")
    m, s2, it = eb([lv[b] for b in used], [vv[b] for b in used])
    n = len(used)
    return {"m": m, "s2": s2, "n": n, "iterations": it, "mu0": m,
            "sigma_mu": math.sqrt(s2) * math.sqrt(1 + 1 / n), "dropped": dropped,
            "levels": {b: lv[b] for b in used}, "mu_var": {b: vv[b] for b in used},
            "omitted_var_mean_t": float(np.mean(omit))}


def round_literal(x) -> float:
    """A level as model.py's LEVEL literal holds it (PE 4.3)."""
    return round(float(x), LITERAL_DECIMALS)


def bundle(fits, excl, cand, *, L, U, level_fits=None, mu0_m=None, single=None,
           literal=False):
    """PB(excl) of PE 4.1 with the level of `cand` (LEVEL_MODES): (prior, hyper,
    report).

    fits[b] for every b of L outside excl (free fits for two or more pairs);
    the subject prior is paiec.prior.from_standings on the standings rows of
    U outside excl, with `included` sorted(L - excl) and `excluded`
    sorted(excl); the widths are hyper_arith's (th2 clipped to ID_CLIP,
    id_cap ID_CAP). The level:
      literal  LEVEL_A3 (C1)
      matched  mu0 = mu0_m, sigma_mu MATCHED_SIGMA_MU, attr_scale ATTR_SCALE (C1m)
      eb       eb_level on level_fits (the fixed-T fits) of L - excl, attr_scale
               ATTR_SCALE (C2); with literal=True mu0 and sigma_mu rounded as
               model.py's LEVEL holds them (the archive's)
      fit      fit_hyper's own levels, which need `single` (RP4).

    Amendment 4a, for every candidate (not for "fit", fit_hyper's own
    arithmetic): the benchmarks of L - excl whose free fit is degenerate
    (degenerate()) are left out of T, the group share, sigma_delta and
    sigma_attr (pooled_meta) and C2's level distribution; their rows still
    enter the standings (the ridge and the identity table) and they stay in
    `included`. C1m's matching leaves them out too (match_set). Amendment 4b:
    the ridge's residual variance s2_res, and so every subject's attribute
    variance u, is pooled without them too (pooled_prior: the prior's
    coefficient covariance and pooled meta variances); sigma_delta is read as
    if they were absent (pooled_meta)."""
    if cand not in LEVEL_MODES:
        raise ValueError(f"cand must be one of {LEVEL_MODES}, got {cand!r}")
    excl = tuple(sorted(set(excl)))
    by = sorted(b for b in L if b not in excl)
    missing = [b for b in by if b not in fits]
    if missing:
        raise PoolError(f"no fit for {missing[:5]}")
    rows = standings_rows(fits, [b for b in U if b not in excl])
    prior = PR.from_standings(rows, ALPHA, by, excl, None) if rows else None
    if cand == "fit" and single is None:
        raise ValueError("cand 'fit' needs `single` for the one-pair benchmarks")
    deg = () if cand == "fit" else tuple(degenerate({b: fits[b] for b in by}))
    meta = pooled_meta(rows, prior, deg) if prior is not None and deg else None
    prior, cov_rep = pooled_prior(rows, prior, meta)
    hyper, rep = hyper_arith(fits, by, prior, excl, single if cand == "fit" else None,
                             deg=deg, meta=meta)
    if hyper.id_cap != ID_CAP:
        raise PoolError(f"id_cap {hyper.id_cap}, not {ID_CAP}")
    rep["cand"], rep["rows"] = cand, len(rows)
    if cov_rep is not None:
        rep["attribute_cov"] = cov_rep
    if cand == "literal":
        hyper = replace(hyper, **LEVEL_A3)
    elif cand == "matched":
        if mu0_m is None:
            raise ValueError("cand 'matched' needs mu0_m")
        hyper = replace(hyper, mu0=float(mu0_m), sigma_mu=MATCHED_SIGMA_MU, attr_scale=ATTR_SCALE)
    elif cand == "eb":
        if level_fits is None:
            raise ValueError("cand 'eb' needs level_fits")
        lev = eb_level(level_fits, prior, [b for b in by if b not in deg])
        mu0, sm = lev["mu0"], lev["sigma_mu"]
        if literal:
            mu0, sm = round_literal(mu0), round_literal(sm)
        hyper = replace(hyper, mu0=mu0, sigma_mu=sm, attr_scale=ATTR_SCALE)
        rep["eb"] = {k: v for k, v in lev.items() if k not in ("levels", "mu_var")}
    return prior, hyper, rep


def candidate_bundle(X, excl, free, fixed, L, U, mu0_m=None, literal=False):
    """The bundle of candidate X (CANDIDATES) for exclusion set excl, from the
    cached fits: free fits for multi-pair benchmarks, the fixed-T fit standing
    in for a one-pair benchmark (only its pair count is read), the fixed-T fits
    as C2's levels."""
    fits = {b: free.get(b, fixed[b]) for b in L}
    return bundle(fits, excl, CANDIDATES[X], L=L, U=U, level_fits=fixed, mu0_m=mu0_m,
                  literal=literal)


def to_bytes(prior, hyper) -> bytes:
    """The bundle's bytes as paiec.prior.save (tools/build_submission.write_bundle)
    writes them."""
    return json.dumps(PR.to_json(prior, hyper), indent=1).encode()


_GH = None


def _gh():
    global _GH
    if _GH is None:
        x, w = np.polynomial.hermite_e.hermegauss(GH_POINTS)
        _GH = (x, w / w.sum())
    return _GH


def theta_moments(prior, hyper, subjects):
    """(m, V) arrays: theta's prior mean and variance without the identity term
    (paiec/hier.py _theta_prior): shift + attr_scale times the attribute mean,
    sigma_theta^2 + u; without an attribute prior shift and sigma_theta^2 +
    sigma_attr^2, as hier falls back."""
    attr = prior is not None and prior.has_attributes
    base = (hyper.shift, hyper.sigma_theta ** 2 + hyper.sigma_attr ** 2)
    m, V = [], []
    for s in subjects:
        mv = base
        if attr:
            try:
                ma, u = prior.attribute({f: _text(s.get(f)) for f in SUBJECT_FIELDS})
                mv = (hyper.shift + hyper.attr_scale * ma, hyper.sigma_theta ** 2 + u)
            except Exception:
                mv = base
        m.append(mv[0])
        V.append(mv[1])
    return np.array(m, float), np.array(V, float)


def _centre_parts(prior, hyper, by_benchmark, which):
    parts = []
    for b in sorted(by_benchmark):
        ss = [s for s in by_benchmark[b]
              if which == "all" or (which == "dated") == bool(_text(s.get("release_date")))]
        if ss:
            parts.append(theta_moments(prior, hyper, ss))
    return parts


def _centre(hyper, parts, mu0):
    x, w = _gh()
    T = hyper.sigma_d ** 2 + hyper.sigma_g ** 2
    vals = []
    for m, V in parts:
        S = np.sqrt(hyper.sigma_mu ** 2 + V + hyper.sigma_delta ** 2 + T)
        eta = mu0 + m[:, None] + S[:, None] * x[None, :]
        vals.append(float(np.mean((1 - hyper.slip) * (1 / (1 + np.exp(-eta))) @ w)))
    return float(np.mean(vals)) if vals else float("nan")


def b0_centre(prior, hyper, by_benchmark, which="all"):
    """(c, benchmarks): the B0 centre of PE 4.2, the mean over benchmarks, each
    once, of the mean over its subjects s of (1 - slip) E_z[sigmoid(mu0 + m(s)
    + sqrt(sigma_mu^2 + V(s) + sigma_delta^2 + T) z)], T = sigma_d^2 +
    sigma_g^2, z on GH_POINTS Gauss-Hermite points; `which` 'all', 'dated'
    or 'undated' (a release_date or none)."""
    parts = _centre_parts(prior, hyper, by_benchmark, which)
    return _centre(hyper, parts, hyper.mu0), len(parts)


def match_mu0(prior, hyper, by_benchmark, target, which="all"):
    """The mu0 at which b0_centre(prior, hyper with that mu0, by_benchmark) equals
    `target`: bisection on MATCH_BRACKET until the bracket is narrower than
    MATCH_TOL; unrounded (C1m rounds it with round_literal)."""
    parts = _centre_parts(prior, hyper, by_benchmark, which)
    lo, hi = MATCH_BRACKET
    if not _centre(hyper, parts, lo) <= target <= _centre(hyper, parts, hi):
        raise PoolError(f"target centre {target} outside the bracket {MATCH_BRACKET}")
    while hi - lo > MATCH_TOL:
        mid = 0.5 * (lo + hi)
        if _centre(hyper, parts, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- what each digest covers (N5) ------------------------------------------------------

#: the bundle arithmetic: no post-lock fix may change it (PE 9, N5)
ARITHMETIC = ("standings_rows", "degenerate", "pooled_meta", "pooled_prior", "match_set",
              "hyper_arith", "eb", "eb_level", "round_literal", "bundle", "candidate_bundle",
              "to_bytes", "_gh", "theta_moments", "_centre_parts", "_centre", "b0_centre",
              "match_mu0")
#: what turns pool files into a fit's pairs: the fits cache is keyed on it
LOADER = ("_text", "_require", "_Resp", "_read_responses", "_responses", "subjects_of",
          "_subject", "items_of", "_item_dicts", "cap_order", "item_rank", "_fit_cells",
          "_fit_pairs", "load_fit_pairs", "fit_one")


def _source_sha(names) -> str:
    h = hashlib.sha256()
    mod = sys.modules[__name__]
    for n in names:
        h.update(f"\x1e{n}\x1f".encode())
        h.update(inspect.getsource(getattr(mod, n)).encode())
    return h.hexdigest()


def source_digest(part) -> str:
    """sha256 of the source of `part`: 'arithmetic' (ARITHMETIC), 'loader'
    (LOADER plus the loader's constants), or 'module' (this file)."""
    if part == "arithmetic":
        return _source_sha(ARITHMETIC)
    if part == "loader":
        consts = json.dumps({k: CONSTANTS[k] for k in ("REVISION", "MIN_ITEMS", "K", "M_P",
                                                       "RESPONSE_COLUMNS", "ITEM_COLUMNS")},
                            sort_keys=True, default=list)
        return hashlib.sha256((_source_sha(LOADER) + consts).encode()).hexdigest()
    if part == "module":
        return _sha256_file(os.path.abspath(__file__))
    raise ValueError(f"part must be arithmetic, loader or module, got {part!r}")


def constants_digest() -> str:
    """sha256 of every constant (CONSTANTS), canonical JSON."""
    return hashlib.sha256(json.dumps(CONSTANTS, sort_keys=True, default=list).encode()).hexdigest()


# --- archive-3's prior, the lists, the archive bundles ---------------------------------

def a3_prior_path() -> str:
    """data/pool_eval/a3_prior.json (impl_plan step 1), else submission/prior.json,
    whichever has archive-3's digest; PoolError if neither has it."""
    for p in (os.path.join(EVAL_DIR, "a3_prior.json"), os.path.join(ROOT, "submission", "prior.json")):
        if os.path.exists(p) and _sha256_file(p) == A3_PRIOR_SHA256:
            return p
    raise PoolError("no copy of archive-3's prior.json with sha256 " + A3_PRIOR_SHA256[:16])


def lists(root=None, jobs=1):
    """(pool_benchmarks, pool_families): the contents of tools/pool_benchmarks.json
    and tools/pool_families.json, names and catalogue keys only."""
    L, U, rep = select(root, jobs)
    E = evaluation_set(L)
    fam = families(root)
    bench = {"revision": REVISION, "rule": SELECTION_RULE, "min_items": MIN_ITEMS,
             "share01_min": SHARE01_MIN, "L": L, "U": U, "E": E, "legacy": list(LEGACY),
             "flagged": list(FLAGGED), "mixed": sorted(b for b in L if rep["benchmarks"][b]["kind"] == "mixed"),
             "mcq_catalogue": sorted(b for b in E if mcq_flag(b, root)), "mcq_regex": MCQ_RE.pattern,
             "counts": {"L": len(L), "U": len(U), "E": len(E)}}
    fams = {"revision": REVISION, "rule": FAMILY_RULE, "benchmarks": fam["benchmarks"],
            "families": fam["families"], "multi_member_families": fam["multi_member_families"],
            "family_of": fam["family_of"], "keys": fam["keys"], "multi": fam["multi"],
            "joining_keys": fam["joining_keys"],
            "multi_in_E": {f: [b for b in m if b in E] for f, m in fam["multi"].items()
                           if sum(b in E for b in m) >= 2}}
    return bench, fams


def write_lists(root=None, out_dir=None, jobs=1) -> dict:
    """Write tools/pool_benchmarks.json and tools/pool_families.json; their sha256."""
    out_dir = out_dir or TOOLS
    bench, fams = lists(root, jobs)
    out = {}
    for name, obj in (("pool_benchmarks.json", bench), ("pool_families.json", fams)):
        data = _json_bytes(obj)
        _write(os.path.join(out_dir, name), data)
        out[name] = hashlib.sha256(data).hexdigest()
    return out


def read_lists(out_dir=None):
    """(pool_benchmarks, pool_families) as committed."""
    out_dir = out_dir or TOOLS
    with open(os.path.join(out_dir, "pool_benchmarks.json")) as f:
        bench = json.load(f)
    with open(os.path.join(out_dir, "pool_families.json")) as f:
        fams = json.load(f)
    return bench, fams


def archive_bundles(free, fixed, L, U, mu0_m, write=True) -> dict:
    """The excl-empty bundles of C1, C1m and C2, as the archive would ship them
    (C2's level rounded to LEVEL's literal), written to
    EVAL_DIR/bundles/archive_<X>.json; {X: (bytes, sha256, level)}."""
    out = {}
    for X in TIE_ORDER:
        prior, hyper, _ = candidate_bundle(X, (), free, fixed, L, U, mu0_m=mu0_m, literal=True)
        data = to_bytes(prior, hyper)
        if write:
            _write(os.path.join(EVAL_DIR, "bundles", f"archive_{X}.json"), data)
        out[X] = (data, hashlib.sha256(data).hexdigest(),
                  {k: getattr(hyper, k) for k in ("mu0", "sigma_mu", "attr_scale")})
    return out


def rp4_legacy(pairs=None, a3_path=None) -> dict:
    """RP4 (PE 9 item 7): this module's bundle code on the legacy five's pairs
    as paiec.data.load_pairs() gives them (items, no caps), against the
    shipped build. Returns, with the BLAS settings it ran under:
      literal_equals_shipped_build  excl empty, LEVEL written over, against
                                    paiec.prior.build + LEVEL in this process
      literal_equals_archive_3      the same bytes against archive-3's prior.json
      shipped_build_equals_archive_3  whether this process's BLAS reproduces
                                    archive-3 at all (archive-3 was built under
                                    OpenBLAS's default threading; one BLAS
                                    thread changes the last bits of both)
      leave_one_out                 {parent: fit-mode bundle == build(pairs, (parent,))}"""
    from paiec.data import load_pairs
    from paiec.official import eligible
    pairs = eligible(load_pairs()) if pairs is None else pairs
    by = PR._by_benchmark(pairs)
    L = sorted(by)
    free = {b: PR.benchmark_fit(ps) for b, ps in by.items() if len(ps) >= 2}
    T = float(np.median([f["item_var"] for f in free.values()]))
    fits = {b: free.get(b) or PR.benchmark_fit(ps, s2e=T) for b, ps in by.items()}
    single = (lambda b, t: PR.benchmark_fit(by[b], s2e=t))
    mine = to_bytes(*bundle(fits, (), "literal", L=L, U=L)[:2])
    p, h = PR.build(pairs, ())
    shipped = to_bytes(p, replace(h, **LEVEL_A3))
    with open(a3_path or a3_prior_path(), "rb") as f:
        a3 = f.read()
    loo = {}
    for parent in L:
        p1, h1, _ = bundle(fits, (parent,), "fit", L=L, U=L, single=single)
        loo[parent] = to_bytes(p1, h1) == to_bytes(*PR.build(pairs, (parent,)))
    return {"benchmarks": L, "pairs": len(pairs), "blas": {v: os.environ.get(v) for v in BLAS_VARS},
            "literal_equals_shipped_build": mine == shipped, "literal_equals_archive_3": mine == a3,
            "shipped_build_equals_archive_3": shipped == a3, "literal_sha256": hashlib.sha256(mine).hexdigest(),
            "leave_one_out": loo}


# --- descriptive parts of the fits stage ------------------------------------------------

def _columns(spec) -> list:
    if spec.date_form != "linear" or spec.harnesses or spec.effort_order:
        raise PoolError("only the default design is named here")
    return (["intercept", "date", "has_date", "log_size", "has_size", "small", "big", "think",
             "harness"] + [f"provider:{p}" for p in spec.providers]
            + [f"effort:{e}" for e in spec.efforts])


def lobo_ridge_shifts(fits, U, subjects_E) -> dict:
    """For each benchmark of U with two or more pairs, the largest change in any
    ridge coefficient (by column name; a column that appears or goes counts
    against 0) and in any E subject's attribute score when its rows leave the
    excl-empty ridge (PE 9 item 3)."""
    from paiec.subjects import attrs, design_row
    rows = standings_rows(fits, U)
    coef0, spec0, _, _ = PR._ridge(rows, ALPHA)
    A = [attrs({f: _text(s.get(f)) for f in SUBJECT_FIELDS}) for s in subjects_E]

    def scores(coef, spec):
        return np.array([design_row(a, spec) @ coef for a in A])
    base = dict(zip(_columns(spec0), coef0.tolist()))
    s0 = scores(coef0, spec0)
    out = {}
    for b in sorted({r[0] for r in rows}):
        rest = [r for r in rows if r[0] != b]
        coef, spec, _, _ = PR._ridge(rest, ALPHA)
        c = dict(zip(_columns(spec), coef.tolist()))
        names = sorted(set(base) | set(c))
        d = {n: c.get(n, 0.0) - base.get(n, 0.0) for n in names}
        top = max(names, key=lambda n: (abs(d[n]), n))
        ds = np.abs(scores(coef, spec) - s0)
        out[b] = {"rows": sum(1 for r in rows if r[0] == b), "max_coef_change": abs(d[top]),
                  "coef": top, "max_score_change": float(ds.max()) if len(ds) else 0.0,
                  "mean_score_change": float(ds.mean()) if len(ds) else 0.0}
    return out


def scales_of_mixed(names, root=None) -> dict:
    """How mixed benchmarks declare their items' scales (PE 3.2, descriptive):
    a separate batched read of item_id and grading_criterion only (never
    through items_of), counting grading_criterion's response_scale by kind and
    number of declared values, and how many declare exactly {0, 1}. Counts
    only: no text is kept or written."""
    import pyarrow.parquet as pq
    out = {}
    for b in sorted(names):
        path = verify(b, "items.parquet", root)
        _require(path, ("item_id", "grading_criterion"))
        cnt, binary, bad, n = defaultdict(int), 0, 0, 0
        pf = pq.ParquetFile(path, buffer_size=READ_BUFFER, pre_buffer=False)
        try:
            for batch in pf.iter_batches(batch_size=ITEM_BATCH, columns=["item_id", "grading_criterion"],
                                         use_threads=False):
                for x in batch.column(1).to_pylist():
                    n += 1
                    try:
                        rs = json.loads(x).get("response_scale") if x else None
                        if isinstance(rs, str):
                            rs = json.loads(rs)
                    except (ValueError, AttributeError):
                        bad += 1
                        continue
                    if not isinstance(rs, dict):
                        cnt["none"] += 1
                        continue
                    vals = rs.get("values")
                    k = len(vals) if isinstance(vals, list) else 0
                    cnt[f"{_text(rs.get('kind'))}:{k}"] += 1
                    try:
                        binary += isinstance(vals, list) and sorted(map(float, vals)) == [0.0, 1.0]
                    except (TypeError, ValueError):
                        pass
        finally:
            pf.close()
        out[b] = {"items": n, "scales": dict(sorted(cnt.items())), "declared_01": int(binary),
                  "unparsed": bad}
    return out


def floor_share(b, root=None, cap=20000) -> dict:
    """The share of the first `cap` items (file order) on which the shipped floor
    is positive, paiec.mcq.floor_of(paiec.hier.mcq_text(content)) (PE 3.7);
    counts only."""
    import pyarrow.parquet as pq
    from paiec.hier import mcq_text
    from paiec.mcq import floor_of
    path = verify(b, "items.parquet", root)
    _require(path, ("content",))
    n = pos = 0
    pf = pq.ParquetFile(path, buffer_size=READ_BUFFER, pre_buffer=False)
    try:
        for batch in pf.iter_batches(batch_size=ITEM_BATCH, columns=["content"], use_threads=False):
            for c in batch.column(0).to_pylist():
                n += 1
                try:
                    pos += float(floor_of(mcq_text(c or ""))) > 0
                except Exception:
                    pass
                if n >= cap:
                    break
            if n >= cap:
                break
    finally:
        pf.close()
    return {"items_read": n, "floor_positive": int(pos), "share": pos / n if n else None,
            "items_total": pf.metadata.num_rows}


def probe_items(b, n, root=None) -> dict:
    """N8: in this process, items_of with content for the first n items of b in
    item_rank's order (the items a capped pair would keep first); CPU, wall and
    peak resident set. Run it in a fresh process (python -m paiec.pool
    --probe-items B N) to read the peak of the read alone."""
    import pyarrow.parquet as pq
    path = verify(b, "items.parquet", root)
    rss0 = _maxrss_mb()
    ids = [str(x) for x in pq.read_table(path, columns=["item_id"]).column(0).to_pylist()]
    rank = item_rank(b, ids)
    want = sorted(ids, key=lambda x: rank[x])[:n]
    c0, w0 = time.process_time(), time.perf_counter()
    got = items_of(b, want, root, content=True)
    c1, w1 = time.process_time(), time.perf_counter()
    chars = sum(len(t) for t, _ in got.values())
    del got
    return {"benchmark": b, "items": len(want), "items_in_table": len(ids), "content_chars": chars,
            "cpu_s": c1 - c0, "wall_s": w1 - w0, "maxrss_mb_before": rss0,
            "maxrss_mb": _maxrss_mb()}


def _probe_fresh(b, n, root=None) -> dict:
    import subprocess
    env = dict(os.environ)
    for v in BLAS_VARS:
        env.setdefault(v, "1")
    r = subprocess.run([sys.executable, "-m", "paiec.pool", "--probe-items", b, str(n),
                        "--root", _root(root)], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise PoolError(f"item probe {b} {n} failed: {r.stderr[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# --- the fits stage (impl_plan step 4, PE 9 item 3) -------------------------------------

N8_KEYS = ("maxrss_mb", "response_cpu", "item_cpu", "load_cpu", "fit_cpu", "pid")


def _levels_at(fit, prior):
    """A fit's level at attribute score 0 under `prior` (fit_hyper's, PE 4.3)."""
    shift = float(np.mean(fit["t"]))
    if prior is not None and prior.has_attributes:
        shift -= float(np.mean([prior.attribute(s)[0] for s in fit["subjects"]]))
    return fit["mu"] + shift


def fits_stage(jobs=4, root=None, out=None, floor=True, probes=("mind2web", "swe_smith"),
               log=print) -> dict:
    """The `fits` stage (impl_plan step 4): the checks, the fits, the archive
    bundles and the records of PE 9 item 3 that need no split. The R4 slopes
    and the run-level recomputations of mu0_m need S and the regimes and are
    experiments/pool_eval.py's. Writes `out` (results/pool_fits.json by
    default) and returns the record."""
    t_start = time.time()
    env = {v: os.environ.get(v) for v in ("PYTHONHASHSEED",) + BLAS_VARS}
    if env["PYTHONHASHSEED"] != "0" or any(env[v] != "1" for v in BLAS_VARS):
        raise PoolError(f"run with PYTHONHASHSEED=0 and every BLAS variable at 1, got {env}")
    rec = {"what": "the fits stage of docs/plans/p2_pool_eval.md (PE 9 item 3)",
           "revision": REVISION, "env": env, "K": K, "M_P": M_P,
           "pool_py": {p: source_digest(p) for p in ("module", "loader", "arithmetic")},
           "constants": constants_digest()}
    msha = manifest_sha256(root)
    if msha != MANIFEST_SHA256:
        raise PoolError(f"manifest sha256 {msha}, the design's is {MANIFEST_SHA256}")
    rec["manifest"] = {"sha256": msha, "fetched_utc": manifest(root).get("fetched_utc"),
                       "items_skipped": manifest(root).get("items_skipped")}

    log("selection: pair tables of every public binary and mixed benchmark")
    L, U, srep = select(root, jobs)
    E = evaluation_set(L)
    if not srep["flagged_match"]:
        raise PoolError(f"the catalogue's flagged set {srep['flagged_by_direction']} is not FLAGGED")
    read = sorted(b for b, v in srep["benchmarks"].items() if "eligible_pairs" in v)
    rec["manifest"]["verified_L"] = verify_all(L, root)
    rec["selection"] = {"L": len(L), "U": len(U), "E": len(E), "expected": EXPECTED,
                        "pairs_E": sum(srep["benchmarks"][b]["eligible_pairs"] for b in E),
                        "pairs_U": sum(srep["benchmarks"][b]["eligible_pairs"] for b in U),
                        "csv": check_selection_csv(L),
                        "out": {b: v["reason"] for b, v in srep["benchmarks"].items() if not v["in"]}}
    rec["csv_check"] = check_csvs(read, root)
    rec["families"] = check_families(families(root))
    rec["lists_sha256"] = {n: hashlib.sha256(_json_bytes(x)).hexdigest()
                           for n, x in zip(("pool_benchmarks.json", "pool_families.json"), lists(root))}

    log(f"L {len(L)}, U {len(U)}, E {len(E)}: fits on {jobs} processes")
    c0 = time.time()
    free, fixed, T_full = fit_all(L, jobs, root, log=log)
    rec["fits_wall_s"] = time.time() - c0
    rec["fits_index_sha256"] = fits_index_sha256()
    rec["T_full"] = T_full
    deg = degenerate(free)
    rec["degenerate"] = deg
    rec["degenerate_rule"] = {
        "threshold": DEGENERATE_ITEM_VAR, "amendment": "4a", "item_var": {b: free[b]["item_var"]
                                                                         for b in deg},
        "largest_other": max((free[b]["item_var"] for b in free if b not in deg), default=None),
        "T_full_with_them": float(np.median([free[b]["item_var"] for b in free])),
        "item_var_over_100": sorted(b for b in free if free[b]["item_var"] > REPORT_ITEM_VAR)}
    shares = [free[b]["share"] for b in sorted(free) if free[b]["share"] is not None
              and b not in deg]
    every_share = [free[b]["share"] for b in sorted(free) if free[b]["share"] is not None]
    rec["group_share"] = {"keyed_fits": len(shares),
                          "median": float(np.median(shares)) if shares else None,
                          "used": min(float(np.median(shares)), PR.G_CAP) if shares else None,
                          "median_with_degenerate": float(np.median(every_share))
                          if every_share else None}
    rec["convergence"] = {
        "free_em_not_converged": sorted(b for b in free if not free[b]["converged"]),
        "free_newton_not_converged": sorted(b for b in free if not free[b]["newton_converged"]),
        "fixed_em_not_converged": sorted(b for b in fixed if not fixed[b]["converged"]),
        "fixed_newton_not_converged": sorted(b for b in fixed if not fixed[b]["newton_converged"])}
    every = list(free.values()) + list(fixed.values())
    rec["cpu_s"] = {"free_fits": sum(f["fit_cpu"] for f in free.values()),
                    "fixed_T_fits": sum(f["fit_cpu"] for f in fixed.values()),
                    "loads": sum(f["load_cpu"] for f in every),
                    "max_worker_rss_mb": max(f["maxrss_mb"] for f in every)}

    log("bundles, mu0_m and the B0-centre table")
    a3p, a3h = PR.load(a3_prior_path())
    c1p, c1h, c1r = candidate_bundle("C1", (), free, fixed, L, U)
    rec["pooled"] = {"degenerate": c1r.get("degenerate"), **(c1r.get("pooled") or {}),
                     "item_var": c1r.get("item_var"), "share": c1r.get("share"),
                     "attribute_cov": c1r.get("attribute_cov")}
    real, ds = subject_sets(E, root)
    # amendment 4b, descriptive: the attribute variance u = x' C x of E's pair subjects
    # under the excl-empty prior (C at the pooled s2_res), and scaled to the all-row s2_res
    ac = c1r.get("attribute_cov") or {}
    uE = [c1p.attribute({f: _text(s.get(f)) for f in SUBJECT_FIELDS})[1]
          for b in sorted(real) for s in real[b]]
    rec["u"] = {"standings_rows": {"pooled": ac.get("u_mean_rows"),
                                   "all_rows": ac.get("u_mean_all_rows")},
                "E_pair_subjects": {"pooled": float(np.mean(uE)) if uE else None,
                                    "all_rows_scaled": float(np.mean(uE)) * ac["s2_res_all_rows"]
                                    / ac["s2_res"] if uE and ac else None, "pairs": len(uE)}}
    ms = match_set(ds, free)          # amendment 4a: the matching leaves them out
    target = b0_centre(a3p, a3h, ms, "all")[0]
    mu0_raw = match_mu0(c1p, c1h, ms, target)
    mu0_m = round_literal(mu0_raw)
    rec["mu0_m"] = {"raw": mu0_raw, "locked": mu0_m, "target_A3_plus_1.25y_all": target,
                    "matching_set": {"benchmarks": len(ms), "pairs": sum(len(v) for v in ms.values()),
                                     "left_out": sorted(set(ds) - set(ms))},
                    "if_matched_on_dated_only": match_mu0(
                        c1p, c1h, ms, b0_centre(a3p, a3h, ms, "dated")[0], "dated"),
                    "if_matched_on_all_of_E": match_mu0(c1p, c1h, ds, b0_centre(a3p, a3h, ds)[0])}
    bundles = archive_bundles(free, fixed, L, U, mu0_m)
    _, c2h, c2r = candidate_bundle("C2", (), free, fixed, L, U, literal=True)
    rec["C2"] = {"literal": {k: getattr(c2h, k) for k in ("mu0", "sigma_mu", "attr_scale")},
                 **c2r["eb"]}
    rec["bundles"] = {X: {"path": os.path.relpath(os.path.join(EVAL_DIR, "bundles",
                                                               f"archive_{X}.json"), ROOT),
                          "sha256": v[1], "bytes": len(v[0]), "level": v[2]}
                      for X, v in bundles.items()}
    arms = {"A3": (a3p, a3h)}
    arms.update({X: PR.from_json(json.loads(v[0])) for X, v in bundles.items()})
    rec["hyper"] = {X: {k: v for k, v in h.to_dict().items() if k not in ("included", "excluded")}
                    for X, (_, h) in arms.items()}
    for name, sets in (("", (real, ds)), ("_matching_set", (match_set(real, free), ms))):
        tab = {}
        for X, (p, h) in arms.items():
            for dk, by in (("real", sets[0]), ("plus_1.25y", sets[1])):
                for which in ("all", "dated", "undated"):
                    c, nb = b0_centre(p, h, by, which)
                    tab.setdefault(X, {})[f"{dk}:{which}"] = c
                    tab[X][f"{dk}:{which}:benchmarks"] = nb
        rec[f"b0_centre{name}"] = tab
        rec[f"b0_centre{name}_minus_A3"] = {X: {k: tab[X][k] - tab["A3"][k] for k in tab["A3"]
                                                if not k.endswith("benchmarks")} for X in TIE_ORDER}
    rec["b0_pairs"] = {"E": sum(len(v) for v in real.values()),
                       "dated": sum(bool(s["release_date"]) for v in real.values() for s in v)}

    log("per-benchmark records")
    rec["benchmarks"] = {}
    for b in L:
        f, g = free.get(b), fixed[b]
        rec["benchmarks"][b] = {
            "kind": srep["benchmarks"][b]["kind"], "flagged": b in FLAGGED, "in_U": b in U,
            "eligible_pairs": srep["benchmarks"][b]["eligible_pairs"], "pairs_kept": g["n_pairs"],
            "items_kept": g["n_items"], "cells": g["n_cells"],
            "free": None if f is None else {k: f[k] for k in (
                "item_var", "share", "keys", "converged", "newton_converged", "iterations",
                "fit_cpu", "load_cpu", "maxrss_mb", "mu", "mu_var")},
            "fixed_T": {k: g[k] for k in ("converged", "newton_converged", "iterations", "fit_cpu",
                                          "load_cpu", "maxrss_mb", "mu", "mu_var", "s2_t")},
            "level_free": None if f is None else _levels_at(f, c1p),
            "level_fixed_T": _levels_at(g, c1p)}
    rec["n8"] = {"fits": {b: {"free": None if b not in free else
                              {k: free[b][k] for k in N8_KEYS},
                              "fixed_T": {k: fixed[b][k] for k in N8_KEYS}}
                          for b in probes if b in fixed},
                 "item_reads": {b: [_probe_fresh(b, n, root) for n in (160, M_P)]
                                for b in probes if b in fixed}}
    log("leave-one-benchmark-out ridge shifts")
    rec["lobo_ridge"] = lobo_ridge_shifts({b: free.get(b, fixed[b]) for b in L}, U,
                                          [s for b in sorted(real) for s in real[b]])
    rec["mixed_scales"] = scales_of_mixed([b for b in L if srep["benchmarks"][b]["kind"] == "mixed"], root)
    rec["catalogue_mcq"] = sorted(b for b in E if mcq_flag(b, root))
    if floor:
        log("floor-active shares of E (content read, counts only)")
        rec["floor_share"] = {b: floor_share(b, root) for b in E}
    rec["wall_s"] = time.time() - t_start
    if out:
        _write(out, _json_bytes(rec))
    return rec


# --- the report (like paiec.prior.main) -------------------------------------------------

def report(root=None, log=print):
    """The excl-empty hyperparameters of every candidate on the cached fits."""
    bench, _ = read_lists()
    free, fixed, T_full = load_fits(root)
    L, U = bench["L"], bench["U"]
    log(f"revision {REVISION[:8]}: L {len(L)}, U {len(U)}, E {len(bench['E'])}; "
        f"T_full {T_full:.3f}; fits index {fits_index_sha256()[:16]}")
    for X in TIE_ORDER:
        path = os.path.join(EVAL_DIR, "bundles", f"archive_{X}.json")
        if not os.path.exists(path):
            log(f"{X}: no archive bundle at {path}")
            continue
        with open(path) as f:
            _, h = PR.from_json(json.load(f))
        d = {k: round(v, 3) for k, v in h.to_dict().items() if k not in ("excluded", "included")}
        log(f"{X} ({CANDIDATES[X]}): {d}; link weight {h.link_weight:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", action="store_true", help="run the fits stage")
    ap.add_argument("--check", action="store_true", help="manifest, selection, families and CSV checks")
    ap.add_argument("--lists", action="store_true", help="write tools/pool_benchmarks.json and pool_families.json")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pool_fits.json"))
    ap.add_argument("--no-floor", action="store_true", help="skip the floor-active shares of E")
    ap.add_argument("--root", default=None)
    ap.add_argument("--index", action="store_true", help="build the eval loader's index for E")
    ap.add_argument("--probe-items", nargs=2, metavar=("B", "N"))
    a = ap.parse_args(argv)
    if a.jobs > 4:
        raise SystemExit("at most 4 processes")
    if a.probe_items:
        print(json.dumps(probe_items(a.probe_items[0], int(a.probe_items[1]), a.root)))
        return
    if a.fit:
        rec = fits_stage(a.jobs, a.root, a.out, not a.no_floor)
        print(json.dumps({k: rec[k] for k in ("T_full", "mu0_m", "bundles", "wall_s")},
                         indent=1, default=str))
        return
    if a.check:
        L, U, rep = select(a.root, a.jobs)
        fam = families(a.root)
        print(json.dumps({"L": len(L), "U": len(U), "E": len(evaluation_set(L)),
                          "flagged_match": rep["flagged_match"],
                          "selection_csv": check_selection_csv(L),
                          "csv": check_csvs(sorted(b for b, v in rep["benchmarks"].items()
                                                   if v.get("eligible_pairs")), a.root),
                          "families": check_families(fam)}, indent=1))
        return
    if a.lists:
        print(json.dumps(write_lists(a.root, jobs=a.jobs), indent=1))
        return
    if a.index:
        L, _, _ = select(a.root, a.jobs)
        got = build_eval_indexes(evaluation_set(L), a.jobs, a.root)
        print(json.dumps({"indexes": len(got), "dir": os.path.join(EVAL_DIR, "index")}))
        return
    report(a.root)


if __name__ == "__main__":
    try:
        main()
    except PoolError as exc:
        sys.exit(f"POOL CHECK FAILED: {exc}")
