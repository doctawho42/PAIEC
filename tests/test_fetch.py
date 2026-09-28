"""paiec/fetch.py without the network: the pinned dataset revision reaches every
download, overrides take precedence in the documented order, failures are
collected, and a local download (when present) was made at the pinned revision."""
import glob
import os
import re
import sys
import types

import pytest

from paiec import fetch as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fake_hub(monkeypatch, calls, fail=()):
    """Install a stand-in huggingface_hub whose hf_hub_download records its
    arguments (and raises for the filenames in `fail`)."""
    def hf_hub_download(repo_id, filename, **kw):
        calls.append(dict(repo_id=repo_id, filename=filename, **kw))
        if filename in fail:
            raise OSError(f"no such file {filename}\nsecond line")
        return os.path.join(kw.get("local_dir", ""), filename)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=hf_hub_download))


def test_revision_is_a_full_commit_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", F.REVISION)


def test_every_download_is_pinned(monkeypatch, tmp_path):
    monkeypatch.delenv("PAIEC_DATA_REVISION", raising=False)
    calls = []
    fake_hub(monkeypatch, calls)
    got, failed = F.fetch(str(tmp_path))
    assert failed == []
    assert len(got) == len(calls) == len(F.BENCHMARKS) * len(F.TABLES)
    for c in calls:
        assert c["repo_id"] == F.REPO
        assert c["repo_type"] == "dataset"
        assert c["local_dir"] == str(tmp_path)
        assert c["revision"] == F.REVISION
    assert {c["filename"] for c in calls} == {f"{b}/{t}.parquet" for b in F.BENCHMARKS
                                              for t in F.TABLES}


def test_overrides(monkeypatch, tmp_path):
    calls = []
    fake_hub(monkeypatch, calls)
    monkeypatch.setenv("PAIEC_DATA_REVISION", "main")
    assert F.revision() == "main"
    F.fetch(str(tmp_path), benchmarks=["matharena"], tables=["items"])
    assert calls[-1]["revision"] == "main"
    F.fetch(str(tmp_path), benchmarks=["matharena"], tables=["items"], rev="abc123")
    assert calls[-1]["revision"] == "abc123"
    monkeypatch.setenv("PAIEC_DATA_REVISION", "")
    assert F.revision() == F.REVISION


def test_failures_are_collected(monkeypatch, tmp_path):
    monkeypatch.delenv("PAIEC_DATA_REVISION", raising=False)
    calls = []
    fake_hub(monkeypatch, calls, fail={"mmdocrag/items.parquet"})
    got, failed = F.fetch(str(tmp_path), benchmarks=["matharena", "mmdocrag"], tables=["items"])
    assert got == ["matharena/items"]
    assert failed == [("mmdocrag/items", "no such file mmdocrag/items.parquet")]


def test_local_download_was_made_at_the_pinned_revision():
    """huggingface_hub writes <local_dir>/.cache/huggingface/download/<path>.metadata
    (revision, etag, timestamp) next to each file it downloads."""
    data = os.environ.get("PAIEC_DATA", os.path.join(ROOT, "data"))
    meta = glob.glob(os.path.join(data, ".cache", "huggingface", "download", "*", "*.parquet.metadata"))
    if not meta:
        pytest.skip("no local measurement-db download")
    revs = {}
    for m in meta:
        with open(m) as f:
            revs[os.path.relpath(m, data)] = f.readline().strip()
    assert set(revs.values()) == {F.REVISION}, revs
