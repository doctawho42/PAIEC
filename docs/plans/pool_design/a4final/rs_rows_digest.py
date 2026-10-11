"""The RS comparator's row files, pinned as one digest (amendment 4, V2).

A3 on RS tasks is read from data/regime_sensitivity_rows/<R>/<i>.json, 520
files under the gitignored data/. This script lists exactly the files
{TUNED,READING,AUDIT,MIXTURE,FLAT,R1B,R1P}/<i>.json (80 a test-like regime,
60 for R1B and R1P), hashes each file's bytes, and hashes the list:

    digest = sha256 of the UTF-8 bytes of
             "".join(f"{path}\\t{sha256}\\n" for path, sha256 in sorted(rows))

with `path` relative to data/regime_sensitivity_rows/ in POSIX form (for
example "AUDIT/0.json"). The `lock` stage of experiments/pool_eval.py records
the same digest by the same rule and `select` re-verifies it before reading
the rows. It also hashes `_compositions.json` (pinned on its own) and the two
other files the lock adds by name (V1, V3).

It opens no row as JSON and reads no score: bytes are hashed, nothing else.
Writes rs_rows_digest.json beside this file.
Run:  PYTHONDONTWRITEBYTECODE=1 python rs_rows_digest.py
"""
import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows")
REGIMES = {"TUNED": 80, "READING": 80, "AUDIT": 80, "MIXTURE": 80, "FLAT": 80,
           "R1B": 60, "R1P": 60}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_digest(root=ROWS):
    rows = []
    for r, n in sorted(REGIMES.items()):
        names = sorted(os.listdir(os.path.join(root, r)))
        want = sorted(f"{i}.json" for i in range(n))
        if names != want:
            raise SystemExit(f"{r}: expected {n} files 0..{n - 1}.json, found {len(names)}")
        rows += [(f"{r}/{name}", sha256_file(os.path.join(root, r, name))) for name in names]
    text = "".join(f"{p}\t{h}\n" for p, h in sorted(rows))
    return len(rows), hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    n, digest = rows_digest()
    client = os.path.join(ROOT, "third_party", "paiec_baseline", "tools",
                          "streaming_ingestion.py")
    clone = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(client)),
                            "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "what": "planning values of the lock entries V1 to V3 (re-computed by the lock)",
        "rs_rows": {"files": n, "rule": "sha256 of sorted 'path\\tsha256\\n' lines",
                    "digest": digest},
        "rs_compositions": sha256_file(os.path.join(ROWS, "_compositions.json")),
        "experiments/__init__.py": sha256_file(os.path.join(ROOT, "experiments", "__init__.py")),
        "streaming_ingestion.py": {"sha256": sha256_file(client), "clone_commit": clone},
    }
    with open(os.path.join(HERE, "rs_rows_digest.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
