"""tools/check_pins.py, offline: the endpoints are faked.

An unchanged dataset and baseline read as 'unchanged' against the pins of
paiec/fetch.py and experiments/baselines_p1.py; a moved one as 'moved'. The Hub
is asked without credentials first and with HF_TOKEN only after a 401 or 403,
and only if HF_TOKEN is set; the token never reaches the record. A refused
GitHub API falls back to git ls-remote. A failed check raises and writes nothing;
each written check is appended to the earlier ones.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_pins", Path(__file__).resolve().parents[1] / "tools" / "check_pins.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

SECRET = "hf_not_a_real_token_0123456789"


def fake_endpoints(ds_sha=C.DATASET_PIN, gh_sha=None, hub_status=200, gh_status=200, need_token=False):
    gh_sha = gh_sha or C.baseline_pin()
    calls = []

    def get(url, headers=None, timeout=30.0):
        calls.append((url, dict(headers or {})))
        if "huggingface.co" in url:
            authed = "Authorization" in (headers or {})
            if need_token and not authed:
                return 401, None
            if hub_status != 200:
                return hub_status, None
            files = [{"rfilename": f"{b}/{t}.parquet"} for b in ("matharena", "swe_rebench")
                     for t in ("items", "response")] + [{"rfilename": "README.md"}]
            return 200, {"sha": ds_sha, "lastModified": "2026-09-18T19:10:30.000Z", "gated": "auto",
                         "private": False, "siblings": files}
        if gh_status != 200:
            return gh_status, None
        if url.endswith("/commits/main"):
            return 200, {"sha": gh_sha, "commit": {"committer": {"date": "2026-09-16T03:57:37Z"}}}
        return 200, {"default_branch": "main", "pushed_at": "2026-09-16T03:57:39Z", "archived": False,
                     "license": None}
    return get, calls


def test_unchanged_pins_read_as_unchanged(tmp_path):
    get, calls = fake_endpoints()
    check = C.run_check(get=get, env={}, clone=str(tmp_path))
    assert check["status"] == "unchanged"
    ds, bl = check["measurement_db"], check["paiec_baseline"]
    assert ds["matches_pin"] and bl["matches_pin"] and ds["auth"] == "anonymous"
    assert ds["benchmarks_with_items_table"] == ["matharena", "swe_rebench"]
    assert bl["method"] == "GitHub REST API" and bl["local_clone"] is None
    assert all("Authorization" not in h for _, h in calls)


def test_a_moved_branch_reads_as_moved(tmp_path):
    get, _ = fake_endpoints(ds_sha="0" * 40)
    check = C.run_check(get=get, env={}, clone=str(tmp_path))
    assert check["status"] == "moved" and not check["measurement_db"]["matches_pin"]


def test_the_token_is_used_only_after_a_refusal_and_never_recorded(tmp_path):
    get, calls = fake_endpoints(need_token=True)
    check = C.run_check(get=get, env={"HF_TOKEN": SECRET}, clone=str(tmp_path))
    assert check["measurement_db"]["auth"].startswith("HF_TOKEN")
    hub = [h for u, h in calls if "huggingface.co" in u]
    assert "Authorization" not in hub[0] and hub[1]["Authorization"] == "Bearer " + SECRET
    assert SECRET not in json.dumps(C.record(check, str(tmp_path / "none.json")))
    get, _ = fake_endpoints(need_token=True)
    with pytest.raises(C.CheckError):
        C.run_check(get=get, env={}, clone=str(tmp_path))


def test_a_refused_github_api_falls_back_to_ls_remote(tmp_path):
    get, _ = fake_endpoints(gh_status=403)
    check = C.run_check(get=get, ls_remote=lambda url, timeout: C.baseline_pin(), env={}, clone=str(tmp_path))
    assert check["paiec_baseline"]["method"].startswith("git ls-remote")
    assert check["paiec_baseline"]["matches_pin"]


def test_checks_are_appended_and_a_failure_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "pins.json"
    get, _ = fake_endpoints()
    first = C.record(C.run_check(get=get, env={}, clone=str(tmp_path)), str(out))
    out.write_text(json.dumps(first))
    second = C.record(C.run_check(get=get, env={}, clone=str(tmp_path)), str(out))
    assert len(second["checks"]) == 2 and second["latest"] == second["checks"][-1]

    def down(*a, **k):
        raise C.CheckError("offline")
    monkeypatch.setattr(C, "run_check", down)
    before = out.read_text()
    assert C.main(["--out", str(out)]) == 1
    assert out.read_text() == before
