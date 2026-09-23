"""Download the core tables of measurement-db.

The dataset is gated: accept the terms on huggingface.co/datasets/aims-foundations/
measurement-db with your account, then either `huggingface-cli login` or set
HF_TOKEN. Only the small core tables are fetched; traces.parquet and assets.parquet
are hundreds of megabytes and nothing here needs them.

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


def fetch(out="data", benchmarks=None, tables=None):
    from huggingface_hub import hf_hub_download
    got, failed = [], []
    for b in benchmarks or BENCHMARKS:
        for t in tables or TABLES:
            try:
                hf_hub_download(REPO, f"{b}/{t}.parquet", repo_type="dataset", local_dir=out)
                got.append(f"{b}/{t}")
            except Exception as e:
                failed.append((f"{b}/{t}", str(e).splitlines()[0][:100]))
    return got, failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    if not (os.environ.get("HF_TOKEN") or os.path.exists(
            os.path.expanduser("~/.cache/huggingface/token"))):
        sys.exit("No Hugging Face credentials found. Log in or set HF_TOKEN; "
                 "the dataset is gated and needs accepted terms.")
    got, failed = fetch(a.out)
    print(f"downloaded {len(got)} tables into {a.out}/")
    for name, err in failed:
        print(f"  failed {name}: {err}")
