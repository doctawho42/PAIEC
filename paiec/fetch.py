"""Download the core tables of measurement-db.

The dataset is gated: accept the terms on huggingface.co/datasets/aims-foundations/
measurement-db with your account, then either `huggingface-cli login` or set
HF_TOKEN. Only the small core tables are fetched; traces.parquet and assets.parquet
are hundreds of megabytes and nothing here needs them.

Every file is fetched at REVISION, the dataset commit that every recorded result
in this repository was computed on (data/.cache/huggingface/download/*.metadata
of the original download names it for all 24 files, and their sha256 match the
recorded etags). The competition page says the collections may grow; set
PAIEC_DATA_REVISION or pass --revision (a commit sha, a branch such as "main",
or a tag) to fetch another state of the dataset.

The data is not vendored into this repository on purpose. It is gated, and
redistributing it would be the wrong thing to do with someone else's terms.
"""
import argparse
import os
import sys

BENCHMARKS = ["matharena", "mmdocrag", "multi_swebench", "real_webagents",
              "researchcodebench", "swe_rebench"]
TABLES = ["response", "items", "subjects", "benchmarks"]
REPO = "aims-foundations/measurement-db"
#: measurement-db commit the recorded results used (downloaded 2026-09-24)
REVISION = "bc8204d811823da849c6686bf124d4ca9f82e4de"


def revision():
    """REVISION, unless PAIEC_DATA_REVISION names another one."""
    return os.environ.get("PAIEC_DATA_REVISION") or REVISION


def fetch(out="data", benchmarks=None, tables=None, rev=None):
    from huggingface_hub import hf_hub_download
    rev = rev or revision()
    got, failed = [], []
    for b in benchmarks or BENCHMARKS:
        for t in tables or TABLES:
            try:
                hf_hub_download(REPO, f"{b}/{t}.parquet", repo_type="dataset", local_dir=out,
                                revision=rev)
                got.append(f"{b}/{t}")
            except Exception as e:
                failed.append((f"{b}/{t}", str(e).splitlines()[0][:100]))
    return got, failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--revision", default=None,
                    help=f"dataset revision (default: $PAIEC_DATA_REVISION or {REVISION})")
    a = ap.parse_args()
    if not (os.environ.get("HF_TOKEN") or os.path.exists(
            os.path.expanduser("~/.cache/huggingface/token"))):
        sys.exit("No Hugging Face credentials found. Log in or set HF_TOKEN; "
                 "the dataset is gated and needs accepted terms.")
    rev = a.revision or revision()
    got, failed = fetch(a.out, rev=rev)
    print(f"downloaded {len(got)} tables into {a.out}/ at revision {rev}")
    for name, err in failed:
        print(f"  failed {name}: {err}")
