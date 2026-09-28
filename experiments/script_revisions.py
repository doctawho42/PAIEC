"""Which version of a script produced each stored result, and what later edits changed.

Two scripts moved into experiments/ were edited after one of their stages had
already written results (the verifier of report draft v1 found this from the
script digests in the results files' `passes`):

* experiments/formative_feedback.py. The preregistration of the pooled reading
  was written at 05:06:59 UTC on 2026-09-28 under script digest d3fe28fd. The
  read stage then ran four times: 05:13:48 (5099ef81), 05:15:57 (321c189e),
  05:19:08 and 05:20:19 (93cb6a84, the file on disk). The stored reading is
  the last.
* experiments/level_audit.py. --stage extra ran under 40ed9e5b; the file has
  changed since.

experiments/script_edits.json holds every edit made in between, as the (old,
new) string replacements they were made with, recovered from the session log
of the lanes that made them. This script

replay   undoes the edits from the files on disk, newest first, and checks each
         intermediate version against the digest recorded for it: in the edit
         log, and in the results file's `passes` (formative_feedback.json) or
         `passes` and `meta` (level_audit.json). Needs no data.
reread   runs formative_feedback.py's read stage under each earlier version
         (d3fe28fd, the preregistration's; 5099ef81 and 321c189e, the first two
         reads') into a temporary copy of the results file, and compares that
         reading with the stored one field by field: the decision (z_mean,
         z_sd, the bias guard, the outcome), every field both report, and the
         fields only one of them reports. Needs what the read stage needs
         (data/subject_side_rows, gitignored; experiments/subject_side.py).

    python experiments/script_revisions.py --stage replay   # instant
    python experiments/script_revisions.py --stage reread   # about 30 s, 0.3 GB

Output: results/script_revisions.json. level_audit.py's extra stage (24
minutes) is not re-run here: the replay shows its edits leave that stage and
the library untouched.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDITS = os.path.join(ROOT, "experiments", "script_edits.json")
OUT = os.path.join(ROOT, "results", "script_revisions.json")
FF = "experiments/formative_feedback.py"
LINKED = ("paiec", "data", "results", "submission")


def digest_text(text):
    """The digest both scripts record: sha256 of the file's bytes, first 16 hex."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def versions(path, revisions):
    """{digest: text} for the file on disk and every earlier version, undoing
    `revisions` (oldest first) newest first. Raises if an edit does not undo
    cleanly or a version's digest is not the one recorded."""
    with open(os.path.join(ROOT, path)) as f:
        text = f.read()
    if digest_text(text) != revisions[-1]["after"]:
        raise ValueError(f"{path} on disk is {digest_text(text)}, the edit log ends at "
                         f"{revisions[-1]['after']}: log the new edit or re-run its stages")
    out = {digest_text(text): text}
    for rev in reversed(revisions):
        for old, new in reversed(rev["edits"]):
            if text.count(new) != 1:
                raise ValueError(f"{path}: an edit of {rev['when_utc']} does not undo cleanly")
            text = text.replace(new, old)
        if digest_text(text) != rev["before"]:
            raise ValueError(f"{path}: undoing {rev['when_utc']} gives {digest_text(text)}, "
                             f"not {rev['before']}")
        out[rev["before"]] = text
    return out


def recorded_digests(entry):
    """Script digests the results file recorded, with the stage and time of each."""
    path = os.path.join(ROOT, entry["results"])
    if not os.path.exists(path):
        return []
    state = load_json(path)
    rows = []
    for p in state.get("passes", []):
        d = p.get("digests", {}).get(FF) or p.get("script_digest")
        rows.append({"stage": p.get("stage") or p.get("command"), "when_utc": p.get("when_utc"),
                     "script_digest": d})
    return rows


def stage_replay(log):
    out = {}
    for path, entry in log["files"].items():
        vs = versions(path, entry["revisions"])
        rec = recorded_digests(entry)
        seen = sorted({r["script_digest"] for r in rec if r["script_digest"]})
        out[path] = {
            "on_disk": entry["revisions"][-1]["after"], "note": entry.get("note", ""),
            "versions_rebuilt": list(vs),
            "revisions": [{k: r[k] for k in ("when_utc", "before", "after", "stage", "what")}
                          for r in entry["revisions"]],
            "passes": rec,
            "recorded_digests_rebuilt": {d: d in vs for d in seen},
        }
    return out


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}/{i}")
    else:
        yield prefix, obj


def compare(stored, other):
    a, b = dict(leaves(stored)), dict(leaves(other))
    common = set(a) & set(b)
    top = lambda keys: sorted({k.split("/")[1] for k in keys})
    return {"leaves": len(b), "common_with_stored": len(common),
            "common_that_differ": sorted(k for k in common if a[k] != b[k]),
            "only_in_stored": {"count": len(set(a) - set(b)), "under": top(set(a) - set(b))},
            "only_in_this_version": {"count": len(set(b) - set(a)), "under": top(set(b) - set(a))}}


def run_read(text, results):
    """formative_feedback.py's read stage under `text`, in a temporary tree whose
    other files are links to this repository; returns the reading it wrote."""
    tmp = tempfile.mkdtemp(prefix="ff_reread_")
    try:
        os.makedirs(os.path.join(tmp, "experiments"))
        for d in LINKED:
            os.symlink(os.path.join(ROOT, d), os.path.join(tmp, d))
        for f in os.listdir(os.path.join(ROOT, "experiments")):
            if f.endswith(".py") and f != "formative_feedback.py":
                os.symlink(os.path.join(ROOT, "experiments", f), os.path.join(tmp, "experiments", f))
        script = os.path.join(tmp, "experiments", "formative_feedback.py")
        with open(script, "w") as f:
            f.write(text)
        out = os.path.join(tmp, "formative_feedback.json")
        shutil.copyfile(os.path.join(ROOT, results), out)
        r = subprocess.run([sys.executable, script, "--stage", "read", "--out", out], cwd=tmp,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
        return load_json(out)["reading"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stage_reread(log):
    entry = log["files"][FF]
    vs = versions(FF, entry["revisions"])
    stored = load_json(os.path.join(ROOT, entry["results"]))["reading"]
    served = {r["before"]: r for r in entry["revisions"]}
    out = {"stored_reading_from": entry["revisions"][-1]["after"], "versions": {}}
    for d, text in vs.items():
        if d == entry["revisions"][-1]["after"]:
            continue
        reading = run_read(text, entry["results"])
        c = compare(stored, reading)
        dec = reading["decision"]
        out["versions"][d] = {
            "followed_by_edit_of": served[d]["when_utc"],
            "decision_equals_stored": dec == stored["decision"],
            "z_mean": dec["z_mean"], "z_sd": dec["z_sd"],
            "bias_guard_trips": dec["bias_guard_trips"], "outcome": dec["outcome"],
            **c}
        old_split = reading.get("run2_b0_excess_by_rate_K=15")
        if old_split:   # the descriptive block the 05:15:31 edit rewrote, as it was
            out["versions"][d]["run2_b0_excess_by_rate_K=15"] = {
                k: v for k, v in old_split.items() if k != "per_pair"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("replay", "reread"), required=True)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    log = load_json(EDITS)
    state = load_json(a.out) if os.path.exists(a.out) else {"passes": []}
    state["about"] = log["about"]
    state[a.stage] = stage_replay(log) if a.stage == "replay" else stage_reread(log)
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    with open(os.path.abspath(__file__), "rb") as f:
        me = hashlib.sha256(f.read()).hexdigest()[:16]
    state["passes"].append({
        "stage": a.stage, "command": " ".join(["python", "experiments/script_revisions.py"] + sys.argv[1:]),
        "when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "commit": head, "script_digest": me,
        "edit_log_digest": digest_text(open(EDITS).read())})
    with open(a.out, "w") as f:
        json.dump(state, f, indent=1)
        f.write("\n")
    if a.stage == "replay":
        for path, v in state["replay"].items():
            print(path, "versions", v["versions_rebuilt"], "recorded digests rebuilt",
                  v["recorded_digests_rebuilt"])
    else:
        for d, v in state["reread"]["versions"].items():
            print(d, "decision equal:", v["decision_equals_stored"], "z_mean", v["z_mean"],
                  "common", v["common_with_stored"], "differ", len(v["common_that_differ"]),
                  "only stored", v["only_in_stored"], "only this", v["only_in_this_version"])


if __name__ == "__main__":
    main()
