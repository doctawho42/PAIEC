"""experiments/script_revisions.py: the edit log rebuilds every script version
that produced a stored result, and the stored re-reads agree with the reading
the report quotes. No data needed."""
import json
import os

import pytest

from experiments import script_revisions as SR


def test_edit_log_rebuilds_every_recorded_version():
    log = SR.load_json(SR.EDITS)
    for path, entry in log["files"].items():
        vs = SR.versions(path, entry["revisions"])       # raises on a stale log or a bad undo
        for rev in entry["revisions"]:
            assert rev["before"] in vs and rev["after"] in vs
        rec = [r["script_digest"] for r in SR.recorded_digests(entry) if r["script_digest"]]
        if not rec:
            continue
        # the stages whose outputs are stored ran under versions the log rebuilds
        if path == SR.FF:
            assert set(rec) <= set(vs)
        else:
            state = SR.load_json(os.path.join(SR.ROOT, entry["results"]))
            extra = [p["script_digest"] for p in state["passes"] if "extra" in p["command"]]
            assert extra and extra[-1] in vs
            assert state["meta"]["script_digest"] == rec[-1] == entry["revisions"][-1]["after"]


def test_versions_refuses_an_edit_that_does_not_undo(tmp_path, monkeypatch):
    (tmp_path / "x.py").write_text("a = 1\nb = 2\n")
    monkeypatch.setattr(SR, "ROOT", str(tmp_path))
    good = [{"when_utc": "t", "before": SR.digest_text("a = 1\nb = 1\n"),
             "after": SR.digest_text("a = 1\nb = 2\n"), "edits": [["b = 1", "b = 2"]]}]
    assert set(SR.versions("x.py", good)) == {good[0]["before"], good[0]["after"]}
    wrong = [dict(good[0], before="0" * 16)]
    with pytest.raises(ValueError):
        SR.versions("x.py", wrong)
    stale = [dict(good[0], after="f" * 16)]
    with pytest.raises(ValueError):
        SR.versions("x.py", stale)


def test_compare_counts_common_and_missing_fields():
    a = {"decision": {"z": 1.0}, "x": [1, 2], "gone": 3}
    b = {"decision": {"z": 1.0}, "x": [1, 5], "new": {"k": 0}}
    c = SR.compare(a, b)
    assert c["common_with_stored"] == 3 and c["common_that_differ"] == ["/x/1"]
    assert c["only_in_stored"] == {"count": 1, "under": ["gone"]}
    assert c["only_in_this_version"] == {"count": 1, "under": ["new"]}


def test_stored_rereads_agree_with_the_reading():
    if not os.path.exists(SR.OUT):
        pytest.skip("no results/script_revisions.json")
    with open(SR.OUT) as f:
        state = json.load(f)
    if "reread" not in state:
        pytest.skip("reread not run")
    vs = state["reread"]["versions"]
    assert {"d3fe28fdda6626ec", "5099ef8179cf67be", "321c189e8104be3c"} <= set(vs)
    for v in vs.values():
        assert v["decision_equals_stored"] and v["common_that_differ"] == []
        assert v["outcome"] == "no candidate: LEVEL stays"
