"""Check the two upstream pins against what is live now, and record the check (R6).

Every recorded result used measurement-db at revision bc8204d8... (paiec/fetch.py
REVISION) and the organisers' baseline repository at commit 82d330dd...
(experiments/baselines_p1.py BASELINE_COMMIT; CLAUDE.md, third_party/). The
report's App G.2 said that whether either has moved "cannot be checked offline".
This tool checks it online and writes results/pins.json:

  measurement_db   the current revision (sha) of the dataset's `main` branch and
                   its last-modified time, from the Hugging Face Hub's public API
                   (GET https://huggingface.co/api/datasets/<repo>/revision/main),
                   with the file count and top-level directories of that
                   revision; compared with the pin
  paiec_baseline   the head of the repository's default branch, from GitHub's
                   public REST API (GET /repos/<repo>, then /commits/<branch>);
                   if the API refuses (rate limit), `git ls-remote <url> HEAD`;
                   compared with the pin, and the local clone's HEAD in
                   third_party/paiec_baseline when it exists

Authentication. Both endpoints are public; the dataset is gated for its files,
not for this metadata. The request is made without credentials first. Only if
the Hub answers 401 or 403 and HF_TOKEN is already set in the environment is it
repeated with that token; the token is never printed or written, and the record
says which way the check was made ("anonymous" or "HF_TOKEN from the
environment"). Nothing else is read from the environment.

Each run appends one check to results/pins.json (`checks`, oldest first) and
sets `latest`; earlier checks are kept. A check that could not reach an endpoint
is printed and not written (exit 1), so an outage never overwrites a record.

Run:  python tools/check_pins.py [--out PATH] [--timeout S]
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec import fetch as F  # noqa: E402  (stdlib only at import; huggingface_hub is lazy)

OUT = os.path.join(ROOT, "results", "pins.json")
DATASET = F.REPO                                   # aims-foundations/measurement-db
DATASET_PIN = F.REVISION
BASELINE_REPO = "aims-foundations/paiec_baseline"
BASELINE_URL = f"https://github.com/{BASELINE_REPO}"
BASELINE_FILE = os.path.join(ROOT, "experiments", "baselines_p1.py")
BASELINE_CLONE = os.path.join(ROOT, "third_party", "paiec_baseline")
HF_API = "https://huggingface.co/api/datasets/{repo}/revision/{rev}"
GH_API = "https://api.github.com/repos/{repo}"
UA = "paiec-check-pins/1.0 (python urllib)"


class CheckError(RuntimeError):
    pass


def baseline_pin(path=BASELINE_FILE):
    """BASELINE_COMMIT as experiments/baselines_p1.py states it (read as text: that
    module loads the data on import)."""
    with open(path) as f:
        m = re.search(r'^BASELINE_COMMIT\s*=\s*"([0-9a-f]{40})"', f.read(), re.M)
    if not m:
        raise CheckError(f"no BASELINE_COMMIT in {path}")
    return m.group(1)


def http_json(url, headers=None, timeout=30.0):
    """(status, parsed JSON or None). Errors with a status are returned, not raised;
    a failure to connect raises CheckError."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise CheckError(f"{url}: {type(e).__name__}: {e}") from None


def check_dataset(get=http_json, env=None, timeout=30.0):
    env = os.environ if env is None else env
    url = HF_API.format(repo=DATASET, rev="main")
    status, body = get(url, None, timeout)
    auth = "anonymous"
    if status in (401, 403) and env.get("HF_TOKEN"):
        status, body = get(url, {"Authorization": "Bearer " + env["HF_TOKEN"]}, timeout)
        auth = "HF_TOKEN from the environment (value not recorded)"
    if status != 200 or not isinstance(body, dict) or not re.fullmatch(r"[0-9a-f]{40}", str(body.get("sha"))):
        raise CheckError(f"{url}: HTTP {status}, auth {auth.split(' ')[0]}")
    files = [s.get("rfilename", "") for s in body.get("siblings") or []]
    tops = sorted({f.split("/", 1)[0] for f in files if "/" in f})
    benches = sorted({f.split("/", 1)[0] for f in files if f.count("/") == 1 and f.endswith("/items.parquet")})
    return {"repo": DATASET, "branch": "main", "endpoint": url, "auth": auth, "http_status": status,
            "sha": body["sha"], "last_modified": body.get("lastModified"),
            "gated": body.get("gated"), "private": body.get("private"),
            "files": len(files), "top_level_dirs": tops, "benchmarks_with_items_table": benches,
            "pinned": DATASET_PIN, "pinned_in": "paiec/fetch.py REVISION",
            "matches_pin": body["sha"] == DATASET_PIN}


def _ls_remote(url, timeout):
    try:
        r = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise CheckError(f"git ls-remote {url}: {e}") from None
    m = re.match(r"([0-9a-f]{40})\s+HEAD", r.stdout)
    if r.returncode != 0 or not m:
        raise CheckError(f"git ls-remote {url}: exit {r.returncode}")
    return m.group(1)


def local_head(path=BASELINE_CLONE):
    if not os.path.isdir(os.path.join(path, ".git")):
        return None
    r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def check_baseline(get=http_json, ls_remote=_ls_remote, timeout=30.0, clone=BASELINE_CLONE):
    pin = baseline_pin()
    api = GH_API.format(repo=BASELINE_REPO)
    hdr = {"Accept": "application/vnd.github+json"}
    status, repo = get(api, hdr, timeout)
    out = {"repo": BASELINE_REPO, "auth": "anonymous", "pinned": pin,
           "pinned_in": "experiments/baselines_p1.py BASELINE_COMMIT; CLAUDE.md"}
    if status == 200 and isinstance(repo, dict) and repo.get("default_branch"):
        branch = repo["default_branch"]
        url = f"{api}/commits/{branch}"
        st2, c = get(url, hdr, timeout)
        if st2 == 200 and isinstance(c, dict) and re.fullmatch(r"[0-9a-f]{40}", str(c.get("sha"))):
            commit = c.get("commit") or {}
            out.update(method="GitHub REST API", endpoint=url, http_status=st2, default_branch=branch,
                       sha=c["sha"], committed=(commit.get("committer") or {}).get("date"),
                       pushed_at=repo.get("pushed_at"), archived=repo.get("archived"),
                       license=(repo.get("license") or {}).get("spdx_id"))
        else:
            status = st2
    if "sha" not in out:
        out.update(method=f"git ls-remote {BASELINE_URL} HEAD (the API answered HTTP {status})",
                   endpoint=BASELINE_URL, sha=ls_remote(BASELINE_URL, timeout))
    out["matches_pin"] = out["sha"] == pin
    head = local_head(clone)
    out["local_clone"] = None if head is None else {"path": os.path.relpath(clone, ROOT), "head": head,
                                                    "matches_pin": head == pin}
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def run_check(get=http_json, ls_remote=_ls_remote, env=None, timeout=30.0, clone=BASELINE_CLONE):
    t0 = time.time()
    when = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    ds = check_dataset(get, env, timeout)
    bl = check_baseline(get, ls_remote, timeout, clone)
    me = os.path.abspath(__file__)
    return {"checked_utc": when, "measurement_db": ds, "paiec_baseline": bl,
            "status": "unchanged" if ds["matches_pin"] and bl["matches_pin"] else "moved",
            "provenance": {"command": " ".join(["python", os.path.relpath(me, ROOT)] + sys.argv[1:]),
                           "script": os.path.relpath(me, ROOT), "script_sha256": sha256(me),
                           "head": git("rev-parse", "HEAD"),
                           "script_status": git("status", "--porcelain", "--", os.path.relpath(me, ROOT)) or "clean",
                           "inputs_sha256": {"paiec/fetch.py": sha256(os.path.join(ROOT, "paiec", "fetch.py")),
                                             "experiments/baselines_p1.py": sha256(BASELINE_FILE)},
                           "python": sys.version.split()[0], "wall_s": round(time.time() - t0, 2)}}


def record(check, path=OUT):
    """Append one check to the record at `path` (kept oldest first)."""
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f).get("checks", [])
    checks = old + [check]
    return {"about": {"what": "the live revision of measurement-db's main branch and the head of the "
                              "organisers' baseline repository, against the pins every recorded result used",
                      "plan": "docs/plans/p2_final_plan.md R6; the report's App G.2",
                      "pins": {"measurement_db": DATASET_PIN, "paiec_baseline": checks[-1]["paiec_baseline"]["pinned"]},
                      "tool": "tools/check_pins.py"},
            "latest": check, "checks": checks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    try:
        check = run_check(timeout=args.timeout)
    except CheckError as e:
        print(f"check not made, nothing written: {e}")
        return 1
    ds, bl = check["measurement_db"], check["paiec_baseline"]
    print(f"{check['checked_utc']}  measurement-db main {ds['sha'][:8]} (last modified {ds['last_modified']}; "
          f"{ds['auth'].split(' ')[0]}) {'= pin' if ds['matches_pin'] else '!= pin ' + ds['pinned'][:8]}")
    print(f"{check['checked_utc']}  paiec_baseline {bl.get('default_branch', 'HEAD')} {bl['sha'][:8]} "
          f"({bl['method'].split(' (')[0]}) {'= pin' if bl['matches_pin'] else '!= pin ' + bl['pinned'][:8]}"
          + (f"; local clone {bl['local_clone']['head'][:8]}" if bl.get("local_clone") else ""))
    out = record(check, args.out)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    os.replace(tmp, args.out)
    print(f"{check['status']}; wrote {os.path.relpath(args.out, ROOT)} ({len(out['checks'])} check(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
