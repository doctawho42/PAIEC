"""Export the per-run row files the report's tables use into a compressed release
directory, with a manifest of sha256s (review P1.17; docs/report/review_v0.md W8
and section 7).

The rows are gitignored under data/ and take CPU hours to regenerate. With them
a reader reproduces the report's checks in minutes: the shipped configuration's
numbers (experiments/ship_confirm.py, about 5 s), P1a's summary
(experiments/regime_sensitivity.py summarise), the acceptance harness's tables
and every covariate read through it, the floored-fit replay, the 14B's
derived tables, and the summaries of the baselines (P1.9) and the pooling
decomposition (P1.11).

Sets (SETS): one deterministic tar.xz each (sorted members, mtime 0, owner 0, mode
0644), so the same rows give the same archive bytes.

  subject_side_rows        data/subject_side_rows        ship_confirm, subject_side --summarise,
                                                         formative_feedback, level_audit,
                                                         script_revisions --stage reread
  regime_sensitivity_rows  data/regime_sensitivity_rows  regime_sensitivity summarise / review
  harness_rows             data/harness_rows             harness table / eval, llm4b_close,
                                                         strong_llm_eval, hidden_state_probe,
                                                         finetune_encoder, gate_and_ci
  harness_rows_legacy      data/harness_rows_legacy      heads_eval --rows legacy, itemcov_eval,
                                                         mcq_floor
  hier_floor               data/hier_floor               hier_floor_replay --stage summary
  strong_llm_eval          data/strong_llm_eval          the 14B's derived tables (features,
                                                         entropy, honest targets, heads' out-of-
                                                         fold predictions); gate_and_ci
  baselines_p1_rows        data/baselines_p1_rows        baselines_p1 summarise (P1.9; it also
                                                         reads regime_sensitivity_rows)
  pooling_decomposition_rows data/pooling_decomposition_rows
                                                         pooling_decomposition summarise /
                                                         recheck (P1.11)

Never exported: measurement-db itself (data/<benchmark>/*.parquet), the item
features and embeddings (data/features), the Kaggle exports, third_party/ and
anything outside data/. The rows are derived from measurement-db, whose terms
(CC-BY-SA, gated) apply to them; the manifest says so.

Usage:
  python tools/export_rows.py                          # export every set to data/release_rows
  python tools/export_rows.py --sets subject_side_rows regime_sensitivity_rows
  python tools/export_rows.py verify [--dir DIR]       # archives and members against MANIFEST.json
  python tools/export_rows.py restore [--dir DIR] [--data DATA] [--force]
                                                       # unpack into data/, checking every member

The default output directory, data/release_rows, is gitignored with the rest of
data/. Exit status 1 if verify or restore finds a mismatch.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PAIEC_DATA", os.path.join(ROOT, "data"))
OUT = os.path.join(DATA, "release_rows")
MANIFEST = "MANIFEST.json"
SUMS = "SHA256SUMS"
XZ_PRESET = 6
#: set -> (directory under data/, what reads it)
SETS = {
    "subject_side_rows": ("subject_side_rows",
                          "experiments/ship_confirm.py; subject_side.py --summarise; formative_feedback.py; "
                          "level_audit.py; script_revisions.py --stage reread"),
    "regime_sensitivity_rows": ("regime_sensitivity_rows", "experiments/regime_sensitivity.py summarise, review"),
    "harness_rows": ("harness_rows",
                     "experiments/harness.py --stage table / eval; llm4b_close.py; strong_llm_eval.py; "
                     "hidden_state_probe.py; finetune_encoder.py; gate_and_ci.py"),
    "harness_rows_legacy": ("harness_rows_legacy",
                            "experiments/heads_eval.py --rows legacy; itemcov_eval.py; mcq_floor.py"),
    "hier_floor": ("hier_floor", "experiments/hier_floor_replay.py --stage summary"),
    "strong_llm_eval": ("strong_llm_eval", "experiments/strong_llm_eval.py (stages after ingest); gate_and_ci.py"),
    "baselines_p1_rows": ("baselines_p1_rows", "experiments/baselines_p1.py summarise (with regime_sensitivity_rows)"),
    "pooling_decomposition_rows": ("pooling_decomposition_rows",
                                   "experiments/pooling_decomposition.py summarise / recheck"),
}
#: never exported, whatever --sets says
EXCLUDED = ("data/<benchmark>/*.parquet (measurement-db itself)", "data/features (item features, embeddings, "
            "Kaggle exports)", "data/.cache", "third_party/", "anything outside data/")
SKIP_NAMES = (".DS_Store",)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def members(src):
    """Sorted relative paths of the regular files under src (no links, no hidden
    junk), as POSIX paths."""
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn in SKIP_NAMES or fn.endswith((".tmp", ".lock")):
                continue
            p = os.path.join(dirpath, fn)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            out.append(os.path.relpath(p, src).replace(os.sep, "/"))
    return sorted(out)


def write_archive(src, prefix, path):
    """A deterministic tar.xz of src's files under prefix/: [{path, sha256, bytes}]."""
    files = []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=XZ_PRESET, format=tarfile.PAX_FORMAT) as tar:
        for rel in members(src):
            with open(os.path.join(src, rel), "rb") as f:
                data = f.read()
            ti = tarfile.TarInfo(f"{prefix}/{rel}")
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tar.addfile(ti, io.BytesIO(data))
            files.append({"path": f"{prefix}/{rel}", "sha256": sha256_bytes(data), "bytes": len(data)})
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)
    return files


def git(*a):
    try:
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def export(out_dir, sets, data_dir=DATA):
    os.makedirs(out_dir, exist_ok=True)
    man = {"what": "per-run row files behind docs/report/draft.md's tables (review P1.17)",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "paiec",
                                                                          "experiments", "submission")),
           "tool": "tools/export_rows.py", "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "format": f"tar.xz (preset {XZ_PRESET}), members sorted, mtime 0, owner 0, mode 0644",
           "licence": ("derived from measurement-db (gated; CC-BY-SA terms as on its dataset page): share alike, "
                       "attribute the organisers' dataset, and accept its terms before using these rows"),
           "excluded": list(EXCLUDED), "sets": {}, "missing": {}}
    for name in sets:
        sub, readers = SETS[name]
        src = os.path.join(data_dir, sub)
        if not os.path.isdir(src) or not members(src):
            man["missing"][name] = f"{os.path.relpath(src, ROOT)} is absent or empty: regenerate it with {readers}"
            print(f"{name}: missing ({src})")
            continue
        t0 = time.time()
        arc = os.path.join(out_dir, f"{name}.tar.xz")
        files = write_archive(src, sub, arc)
        man["sets"][name] = {"source": f"data/{sub}", "read_by": readers, "archive": os.path.basename(arc),
                             "archive_sha256": sha256_file(arc), "archive_bytes": os.path.getsize(arc),
                             "files": len(files), "bytes_uncompressed": sum(f["bytes"] for f in files),
                             "members": files}
        print(f"{name}: {len(files)} files, {man['sets'][name]['bytes_uncompressed'] / 2 ** 20:.1f} MB -> "
              f"{man['sets'][name]['archive_bytes'] / 2 ** 20:.1f} MB in {time.time() - t0:.0f}s")
    man["how_to"] = [
        "accept measurement-db's terms; clone the code at git_head",
        "python tools/export_rows.py verify --dir THIS_DIR",
        "python tools/export_rows.py restore --dir THIS_DIR      # unpacks into data/",
        "python experiments/ship_confirm.py                      # the shipped config's numbers, ~5 s",
        "python experiments/regime_sensitivity.py summarise      # P1a's summary from its rows",
        "python experiments/harness.py --stage show              # the gate table (results/ is in git)",
        "python experiments/gate_and_ci.py --stage single        # swe_rebench, ~5 s",
        "python experiments/baselines_p1.py summarise            # P1.9's summary from its rows and P1a's",
        "python experiments/pooling_decomposition.py summarise   # P1.11's summary from its rows, ~3 s",
    ]
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(man, f, indent=1)
    with open(os.path.join(out_dir, SUMS), "w") as f:
        for name, s in man["sets"].items():
            f.write(f"{s['archive_sha256']}  {s['archive']}\n")
    return man


def verify(out_dir):
    """Problems found ([] if none): every archive's sha256, and every member's
    bytes and sha256, against the manifest."""
    man = json.load(open(os.path.join(out_dir, MANIFEST)))
    bad = []
    for name, s in man["sets"].items():
        arc = os.path.join(out_dir, s["archive"])
        if not os.path.exists(arc):
            bad.append(f"{name}: archive {s['archive']} missing")
            continue
        if sha256_file(arc) != s["archive_sha256"]:
            bad.append(f"{name}: archive sha256 differs")
        want = {m["path"]: m for m in s["members"]}
        seen = set()
        with tarfile.open(arc, "r:xz") as tar:
            for ti in tar:
                if not ti.isfile():
                    continue
                data = tar.extractfile(ti).read()
                m = want.get(ti.name)
                if m is None:
                    bad.append(f"{name}: unexpected member {ti.name}")
                elif m["sha256"] != sha256_bytes(data) or m["bytes"] != len(data):
                    bad.append(f"{name}: member {ti.name} differs")
                seen.add(ti.name)
        for p in sorted(set(want) - seen):
            bad.append(f"{name}: member {p} missing from the archive")
    return bad


def restore(out_dir, data_dir=DATA, force=False, sets=None):
    """Unpack into data_dir, checking each member's sha256 before writing. An
    existing file with other bytes is kept unless force. Returns problems."""
    bad = verify(out_dir)
    if bad:
        return bad
    man = json.load(open(os.path.join(out_dir, MANIFEST)))
    for name, s in man["sets"].items():
        if sets and name not in sets:
            continue
        want = {m["path"]: m for m in s["members"]}
        with tarfile.open(os.path.join(out_dir, s["archive"]), "r:xz") as tar:
            for ti in tar:
                if not ti.isfile():
                    continue
                rel = os.path.normpath(ti.name)
                if rel.startswith("..") or os.path.isabs(rel):
                    bad.append(f"{name}: unsafe member {ti.name}")
                    continue
                data = tar.extractfile(ti).read()
                if sha256_bytes(data) != want[ti.name]["sha256"]:
                    bad.append(f"{name}: member {ti.name} differs")
                    continue
                dest = os.path.join(data_dir, rel)
                if os.path.exists(dest) and sha256_file(dest) != want[ti.name]["sha256"] and not force:
                    bad.append(f"{name}: {dest} exists with other bytes (use --force)")
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp = dest + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, dest)
        print(f"{name}: restored {s['files']} files into {os.path.join(data_dir, s['source'].split('/', 1)[-1])}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", default="export", choices=("export", "verify", "restore"))
    ap.add_argument("--dir", default=OUT, help="the release directory (default data/release_rows)")
    ap.add_argument("--data", default=DATA, help="where the rows live / are restored to (default data/)")
    ap.add_argument("--sets", nargs="*", choices=tuple(SETS), help="default: every set")
    ap.add_argument("--force", action="store_true", help="restore: overwrite files with other bytes")
    a = ap.parse_args(argv)
    if a.action == "export":
        export(a.dir, a.sets or list(SETS), a.data)
        bad = verify(a.dir)
    elif a.action == "verify":
        bad = verify(a.dir)
    else:
        bad = restore(a.dir, a.data, a.force, a.sets)
    for b in bad:
        print("MISMATCH", b, file=sys.stderr)
    print("ok" if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
