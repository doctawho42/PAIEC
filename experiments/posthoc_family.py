"""The post-hoc family (R5): maps of hier-ship's stored predictions fitted leave
one parent out, update-speed and B0 calibrations, and budget-wise switching
between configurations. All of them null or negative.

docs/plans/p2_final_plan.md (section 2's last bullet and R5) and the v2 outline
(T4 row 6, App F.8) cite one family of negative results: every cheap lever
measured on 2026-10-06 on top of hier-ship's predictions. The measurements were
exploratory scripts in a session scratch directory (posthoc_probe.py,
upd_probe.py, upd_probe2.py, b0_probe.py, info/damp_probe.py and
budget_switch_probe.py). This script is their port: it recomputes every number
from files in the repository and checks each against the value the scratch
recorded (`scratch` in the results file). It selects, tunes and ships nothing.

Every lever is post hoc on hier-ship's stored per-item predictions at every
checkpoint, so hier is not refitted. A map is fitted per budget on the
appearances of the other parents and scored on the held-out parent: leave one
parent out (LOPO). As in the scratch, 'lopo' fits on the test-like (TL) and
mix/whole appearances of the other parents (the harness's FIT_ON), 'lopo4' on
the appearances of all four regimes of the other parents, and 'ins' fits and
scores in sample per regime (a ceiling, not a lever). Public runs are scored
with the fit that leaves out the target's benchmark; for swe_rebench, which has
no test-like parent, every parent is used.

Blocks (keys of results/posthoc_family.json), each the port of one probe:

  maps                posthoc_probe.py: per budget k, the temperature
                      (tonly: logit q = b_k logit p), the affine logit map
                      (temp: a_k + b_k logit p), the isotonic map (iso), blends
                      q = (1 - w_k) p + w_k s_k with Smooth-cal (blend: s_k the
                      Beta(0.5, 1.5) mean of the pair's own first k labels) and
                      with Smooth (blend22: Beta(2, 2)), and the run-level shift
                      (runshift: logit q = logit p + c_k R, R the label-weighted
                      mean over the run's other pairs of the logit of their
                      smoothed own-label rate minus the logit of their mean B0
                      prediction); and `calibration`, the in-sample affine map
                      per regime and budget (the ceiling and the B0 offsets)
  update_speed        upd_probe.py: damping toward the item's own B0
                      prediction, logit q_k = logit p_0 + lambda_k (logit p_k -
                      logit p_0) (upd), on the probability scale (updp), and a
                      blend with a Beta-binomial learner anchored on hier's B0
                      pair mean (bba); modes ins, lopo, lopo4; fit weights raw
                      and hidden-v; each scored raw and hidden-v; and
                      `baseline`, hier-ship's excess over the pair-rate oracle
  update_conditional  upd_probe2.py: lambda by the pair's own label pattern at
                      k = 3, 7 (pat: unanimous or mixed), the within-pair spread
                      kappa_k (spr: logit q = logit pbar_k + kappa_k (logit p -
                      logit pbar_k)), lambda scaled by the run's other pairs'
                      label dispersion (disp)
  b0_stretch          b0_probe.py, at B0 only: the between-pair spread about a
                      fixed centre (bw), the within-pair spread (wi), both with
                      the centre fitted (bwa); `describe`, the in-sample slopes
                      by regime and parent
  damping             info/damp_probe.py: lambda_k as upd, bounded to [0.3,
                      1.6] and fitted with run weights (plain) or hidden-v
                      weights (vweighted_fit), scored with equal and hidden-v
                      appearance weights
  budget_switch       budget_switch_probe.py: per budget, switching between the
                      11 configurations scored on the regime-sensitivity rows
                      (predict can read the budget from the pair's own label
                      count, so a switch is a legal predictor); in sample per
                      regime, and the robust schedule that minimises the worst
                      regime, chosen on all seven regimes at once (so
                      optimistic)
  family              one row per lever for the report (T4 row 6): the LOPO
                      reading on TL+mix, the best nested delta over every LOPO
                      variant, the per-fold parameter ranges ("the optimum
                      belongs to the parent")

Weights and SEs, as in each probe. A delta is ALC(lever) - ALC(hier-ship), the
mean over runs of the run's mean pair delta (pairs weighted 1/n); negative is
better. The SE is a (parent, subject) cluster bootstrap with the fit held fixed
(a lower bound): 2,000 resamples for `maps`, 1,000 elsewhere. Fit weights
differ between probes and are kept: posthoc_probe.py and damp_probe.py weigh
each run 1 (so the 300 TL runs outweigh the 150 mix runs two to one),
upd_probe.py weighs each regime 1 ('w_raw'). The hidden-v weights give each
regime's appearances the pooled formative composition in v = B0 - ECE0^2 (bins
below 0.10, 0.10 to 0.20, above 0.20); upd_probe.py's replica shares are
weighted 1/n, damp_probe.py's are counts. These shares are the only use of the
formative tables: 26 v values pooled into three bins, no id read or written.

Random streams. Each block uses the generator, seed and draw order of the probe
it ports (default_rng(0) per probe process, consumed in its summary order;
damp_probe.py a fresh default_rng(1) per cell), so the numbers reproduce bit for
bit. One quirk is kept: posthoc_probe.py's own-label smoother divides by k + n0
even when a pair holds fewer than k labels (4 of 874 benchmark-first
appearances, at B31 only).

Inputs: data/harness_rows/{tl,mix,r1b,r1p}/*.pkl (experiments/harness.py
--stage collect: hier-ship at LEVEL over LOPO priors, every checkpoint's
per-item predictions, library 3f75a549673aae6a, the archive-3 code;
gitignored), data/regime_sensitivity_rows/<regime>/*.json
(experiments/regime_sensitivity.py, seed 11; gitignored) and results/formative/
run{1,2,3}.txt (B0 sections only, for the v shares).

Run:  python experiments/posthoc_family.py [--scratch DIR] [--out PATH]
--scratch DIR names the session scratch directory that holds the probes' JSON
outputs; with it every stored number is compared leaf by leaf. Without it the
embedded scratch values (the probes' printed tables) are still checked. Writes
results/posthoc_family.json only if every required check passes. A few minutes
on one CPU; nothing is parallel.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "results", "posthoc_family.json")
HARNESS_ROWS = os.path.join(ROOT, "data", "harness_rows")
RS_ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows")
TABLES = {r: os.path.join(ROOT, "results", "formative", f"run{r}.txt") for r in (1, 2, 3)}

W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
BUD = (0, 1, 3, 7, 15, 31)
REGS = ("tl", "mix", "r1b", "r1p")
FIT_ON = ("tl", "mix")
EPS = 1e-6
V_EDGES = (0.10, 0.20)
DAMP_BINS = [0.0, 0.10, 0.20, 0.2501]
RS_REGIMES = ["TUNED", "READING", "AUDIT", "MIXTURE", "FLAT", "R1B", "R1P"]
RS_CONFIGS = ("ship", "aggr", "eb_fit", "eb_adapt", "eb_ship", "wide35", "wide50", "legacy",
              "smooth", "smcal", "onepl")
GATE_TL = -0.002          # the harness gate's test-like bar (report section 4.4)
GUARD_PUBLIC = 0.001      # the gate's public-loss guard
NEG_MIN = 0.0001          # a test-like loss below this is read as zero (the bar is 20 times larger)
SEEDS = {"maps": 0, "update_speed": 0, "update_conditional": 0, "b0_stretch": 0,
         "damping": "1, fresh per cell"}


# --- shared ----------------------------------------------------------------------------------------

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def brier(K, N, p):
    return float(((K * (1 - p) ** 2 + (N - K) * p ** 2).sum()) / N.sum())


def load_apps(rows_dir=HARNESS_ROWS):
    """One dict per pair appearance, in the probes' order (regime, run, slot)."""
    apps, libs = [], set()
    for reg in REGS:
        files = sorted(glob.glob(os.path.join(rows_dir, reg, "*.pkl")),
                       key=lambda f: int(os.path.basename(f)[:-4]))
        for f in files:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            libs.add(d.get("lib_digest"))
            n = len(d["slots"])
            for j, s in enumerate(d["slots"]):
                apps.append(dict(reg=reg, run=d["run"], n=n, j=j, parent=s["parent"],
                                 bench=s["bench"], sid=s["sid"], K=s["ev_K"].astype(float),
                                 N=s["ev_N"].astype(float), P=np.asarray(s["ev_p"], float),
                                 y=np.asarray(s["acq_y"], float)))
    return apps, sorted(x for x in libs if x)


def fresh(base):
    """A copy of every appearance dict (arrays shared, never written), so blocks do not
    see each other's keys."""
    return [dict(a) for a in base]


def hidden_v(paths=None):
    """v = B0 - ECE0^2 of every pair in the organisers' tables (B0 section only), as
    upd_probe.hidden_shares parsed them."""
    vs = []
    for r in (1, 2, 3):
        with open((paths or TABLES)[r]) as f:
            txt = f.read()
        blk = re.split(r"Label budget: (\d+)", txt)
        for j in range(1, len(blk), 2):
            if int(blk[j]) != 0:
                continue
            for line in blk[j + 1].splitlines():
                m = re.match(r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", line)
                if m:
                    b, e = float(m.group(4)), float(m.group(5))
                    vs.append(b - e * e)
            break
    return np.array(vs)


def vbin(v):
    return 0 if v < V_EDGES[0] else (1 if v < V_EDGES[1] else 2)


def _bootstrap(S, Wc, rng, nboot):
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(S), len(S))
        out.append(S[pick].sum() / Wc[pick].sum())
    return out


# --- maps: posthoc_probe.py ------------------------------------------------------------------------

def pp_smooth(y, k, n0, m0):
    return (y[:k].sum() + n0 * m0) / (k + n0)


def pp_features(apps):
    """The run-level surprise R from the run's other pairs, per budget."""
    by_run = {}
    for a in apps:
        by_run.setdefault((a["reg"], a["run"]), []).append(a)
    for a in apps:
        a["p0bar"] = float((a["P"][0] * a["N"]).sum() / a["N"].sum())
    for grp in by_run.values():
        for a in grp:
            R = np.zeros(6)
            for bi, k in enumerate(BUD):
                if k == 0:
                    continue
                num = den = 0.0
                for b in grp:
                    if b is a:
                        continue
                    kk = min(k, len(b["y"]))
                    if kk == 0:
                        continue
                    rate = (b["y"][:kk].sum() + 0.5) / (kk + 1.0)
                    num += kk * (logit(rate) - logit(b["p0bar"]))
                    den += kk
                R[bi] = num / den if den > 0 else 0.0
            a["R"] = R


def pp_stack(apps, bi):
    """Items of every appearance at budget bi: p, K, N and the weight per row (each run
    weighs 1, each pair 1/n of it, its responses equally)."""
    ps, Ks, Ns, ws, sm, sm22, Rs = [], [], [], [], [], [], []
    k = BUD[bi]
    for a in apps:
        m = len(a["K"])
        w = 1.0 / a["n"] / a["N"].sum()
        ps.append(a["P"][bi]); Ks.append(a["K"]); Ns.append(a["N"])
        ws.append(np.full(m, w))
        sm.append(np.full(m, pp_smooth(a["y"], k, 2.0, 0.25)))
        sm22.append(np.full(m, pp_smooth(a["y"], k, 4.0, 0.5)))
        Rs.append(np.full(m, a["R"][bi]))
    cat = np.concatenate
    return dict(p=cat(ps), K=cat(Ks), N=cat(Ns), w=cat(ws), sm=cat(sm), sm22=cat(sm22), R=cat(Rs))


def pp_loss(q, S):
    return float((S["w"] * (S["K"] * (1 - q) ** 2 + (S["N"] - S["K"]) * q ** 2)).sum())


def pp_apply(kind, theta, S):
    p = S["p"]
    if kind == "temp":
        return sig(theta[0] + theta[1] * logit(p))
    if kind == "tonly":
        return sig(theta[0] * logit(p))
    if kind == "blend":
        return (1 - theta[0]) * p + theta[0] * S["sm"]
    if kind == "blend22":
        return (1 - theta[0]) * p + theta[0] * S["sm22"]
    if kind == "runshift":
        return sig(logit(p) + theta[0] * S["R"])
    raise ValueError(kind)


def pp_fit(kind, S, bi):
    if kind == "temp":
        x0 = [0.0, 1.0]
    elif kind == "tonly":
        x0 = [1.0]
    else:
        x0 = [0.0]
    if kind == "runshift" and BUD[bi] == 0:
        return np.array([0.0])

    def f(th):
        return pp_loss(pp_apply(kind, th, S), S)
    if kind in ("blend", "blend22"):
        grid = np.linspace(0, 1, 101)
        vals = [f([g]) for g in grid]
        return np.array([grid[int(np.argmin(vals))]])
    r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-12, maxiter=4000))
    return r.x


def pp_iso_fit(S):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    ir.fit(S["p"], S["K"] / S["N"], sample_weight=S["w"] * S["N"])
    return ir


def pp_evaluate(apps, kinds):
    tr_pool = [a for a in apps if a["reg"] in FIT_ON]
    res = {}
    base_alc = np.array([sum(W6[bi] * brier(a["K"], a["N"], a["P"][bi]) for bi in range(6)) for a in apps])
    for a, v in zip(apps, base_alc):
        a["alc0"] = v
    for kind in kinds:
        diff = np.zeros(len(apps))
        bud_diff = np.zeros((len(apps), 6))
        params = {}
        for hp in sorted(set(a["parent"] for a in apps)):
            train = [a for a in tr_pool if a["parent"] != hp]
            tgt_idx = [i for i, a in enumerate(apps) if a["parent"] == hp]
            if not tgt_idx:
                continue
            tgt = [apps[i] for i in tgt_idx]
            Q, th_all = [], []
            for bi in range(6):
                Str, Ste = pp_stack(train, bi), pp_stack(tgt, bi)
                if kind == "iso":
                    Q.append(pp_iso_fit(Str).predict(Ste["p"]))
                    th_all.append(None)
                else:
                    th = pp_fit(kind, Str, bi)
                    th_all.append([float(x) for x in th])
                    Q.append(pp_apply(kind, th, Ste))
            params[hp] = th_all
            alc = np.zeros(len(tgt))
            for bi in range(6):
                off = 0
                for t, a in enumerate(tgt):
                    m = len(a["K"])
                    bq = brier(a["K"], a["N"], Q[bi][off:off + m])
                    b0 = brier(a["K"], a["N"], a["P"][bi])
                    bud_diff[tgt_idx[t], bi] = bq - b0
                    alc[t] += W6[bi] * bq
                    off += m
            for t, i in enumerate(tgt_idx):
                diff[i] = alc[t] - apps[i]["alc0"]
        res[kind] = dict(diff=diff, bud=bud_diff, params=params)
    return res


def pp_summarise(apps, res, rng):
    out = {}
    for kind, r in res.items():
        out[kind] = {"params": r["params"]}
        for reg in REGS:
            idx = [i for i, a in enumerate(apps) if a["reg"] == reg]
            sub = [apps[i] for i in idx]
            d = r["diff"][idx]
            w = np.array([1.0 / a["n"] for a in sub])
            nruns = len(set(a["run"] for a in sub))
            est = float((w * d).sum() / nruns)
            keys = [(a["parent"], a["sid"]) for a in sub]
            uk = sorted(set(keys))
            kix = {k: t for t, k in enumerate(uk)}
            S = np.zeros(len(uk)); Wc = np.zeros(len(uk))
            for t, a in enumerate(sub):
                S[kix[keys[t]]] += w[t] * d[t]; Wc[kix[keys[t]]] += w[t]
            se = float(np.std(_bootstrap(S, Wc, rng, 2000)))
            pp = {}
            for par in sorted(set(a["parent"] for a in sub)):
                ii = [t for t, a in enumerate(sub) if a["parent"] == par]
                pp[par] = float((w[ii] * d[ii]).sum() / w[ii].sum())
            bud = r["bud"][idx]
            out[kind][reg] = dict(delta=est, cluster_se=se,
                                  base_alc=float((w * np.array([a["alc0"] for a in sub])).sum() / nruns),
                                  per_parent=pp,
                                  brier_by_budget=[float((w * bud[:, bi]).sum() / nruns) for bi in range(6)])
    return out


def pp_calibration(apps):
    """Per regime and budget: mean prediction and outcome (response-weighted) and the
    in-sample affine logit map with its gain (a ceiling)."""
    tab = {}
    for reg in REGS:
        sub = [a for a in apps if a["reg"] == reg]
        nruns = len(set(a["run"] for a in sub))
        rows = []
        for bi in range(6):
            S = pp_stack(sub, bi)
            mp = float((S["w"] * S["N"] * S["p"]).sum() / (S["w"] * S["N"]).sum())
            my = float((S["w"] * S["K"]).sum() / (S["w"] * S["N"]).sum())
            th = pp_fit("temp", S, bi)
            rows.append(dict(budget=BUD[bi], mean_p=mp, mean_y=my, a=float(th[0]), b=float(th[1]),
                             gain_in_sample=float(pp_loss(pp_apply("temp", th, S), S) - pp_loss(S["p"], S))
                             / nruns))
        tab[reg] = rows
    return tab


PP_KINDS = ["tonly", "temp", "blend", "blend22", "runshift", "iso"]


def block_maps(base):
    apps = fresh(base)
    pp_features(apps)
    res = pp_evaluate(apps, PP_KINDS)
    summary = pp_summarise(apps, res, np.random.default_rng(0))
    return {"summary": summary, "calibration": pp_calibration(apps), "n_appearances": len(apps)}


# --- update_speed: upd_probe.py --------------------------------------------------------------------

def u_prepare(base, sh):
    """upd_probe.load + set_weights + the per-budget Brier and ALC of hier-ship."""
    apps = fresh(base)
    for a in apps:
        o = a["K"].sum() / a["N"].sum()
        a["v"] = o * (1 - o)
    for reg in REGS:
        sub = [a for a in apps if a["reg"] == reg]
        nruns = len(set(a["run"] for a in sub))
        raw = np.array([1.0 / a["n"] for a in sub])
        bins = np.array([vbin(a["v"]) for a in sub])
        rep = np.array([raw[bins == b].sum() for b in range(3)]) / raw.sum()
        for a, w, b in zip(sub, raw, bins):
            a["w_raw"] = w / nruns
            a["w_hv"] = w / nruns * sh[b] / rep[b]
            a["bin"] = int(b)
    for a in apps:
        bud = np.array([float((a["K"] * (1 - a["P"][bi]) ** 2 + (a["N"] - a["K"]) * a["P"][bi] ** 2).sum()
                              / a["N"].sum()) for bi in range(6)])
        a["bud0"] = bud
        a["alc0"] = float(W6 @ bud)
    return apps


def u_stack(apps, bi, wkey):
    k = BUD[bi]
    out = {x: [] for x in ("p", "p0", "K", "N", "w", "bb_s", "bb_k", "pbar0")}
    for a in apps:
        m = len(a["K"])
        out["p"].append(a["P"][bi]); out["p0"].append(a["P"][0])
        out["K"].append(a["K"]); out["N"].append(a["N"])
        out["w"].append(np.full(m, a[wkey] / a["N"].sum()))
        kk = min(k, len(a["y"]))
        out["bb_s"].append(np.full(m, a["y"][:kk].sum())); out["bb_k"].append(np.full(m, float(kk)))
        out["pbar0"].append(np.full(m, (a["P"][0] * a["N"]).sum() / a["N"].sum()))
    return {x: np.concatenate(v) for x, v in out.items()}


def u_loss(q, S):
    return float((S["w"] * (S["K"] * (1 - q) ** 2 + (S["N"] - S["K"]) * q ** 2)).sum())


def u_apply(kind, th, S):
    if kind == "upd":
        return sig(logit(S["p0"]) + th[0] * (logit(S["p"]) - logit(S["p0"])))
    if kind == "updp":
        return np.clip(S["p0"] + th[0] * (S["p"] - S["p0"]), EPS, 1 - EPS)
    if kind == "bba":
        w, n0 = th
        bb = (n0 * S["pbar0"] + S["bb_s"]) / (n0 + S["bb_k"])
        return (1 - w) * S["p"] + w * bb
    raise ValueError(kind)


def u_fit(kind, S, bi):
    if BUD[bi] == 0:
        return [1.0] if kind != "bba" else [0.0, 2.0]
    if kind in ("upd", "updp"):
        r = minimize_scalar(lambda t: u_loss(u_apply(kind, [t], S), S), bounds=(0.0, 2.5), method="bounded",
                            options=dict(xatol=1e-4))
        return [float(r.x)]
    best = None
    for n0 in (0.5, 1, 2, 4, 8, 16):
        r = minimize_scalar(lambda t: u_loss(u_apply(kind, [t, n0], S), S), bounds=(0.0, 1.0),
                            method="bounded", options=dict(xatol=1e-4))
        if best is None or r.fun < best[0]:
            best = (r.fun, [float(r.x), float(n0)])
    return best[1]


def u_per_app_alc(apps, Q):
    out = np.zeros(len(apps)); bud = np.zeros((len(apps), 6))
    for bi in range(6):
        off = 0
        for t, a in enumerate(apps):
            m = len(a["K"]); q = Q[bi][off:off + m]; off += m
            b = float((a["K"] * (1 - q) ** 2 + (a["N"] - a["K"]) * q ** 2).sum() / a["N"].sum())
            bud[t, bi] = b; out[t] += W6[bi] * b
    return out, bud


def u_run_mode(apps, kind, mode, wfit, stack, fit, apply):
    """Per appearance (delta ALC, delta Brier by budget) and the fitted parameters."""
    d = np.zeros(len(apps)); db = np.zeros((len(apps), 6)); params = {}
    idx_of = {id(a): i for i, a in enumerate(apps)}
    if mode == "ins":
        groups = [(reg, [a for a in apps if a["reg"] == reg], [a for a in apps if a["reg"] == reg])
                  for reg in REGS]
    else:
        pool = [a for a in apps if (a["reg"] in FIT_ON if mode == "lopo" else True)]
        groups = [(hp, [a for a in pool if a["parent"] != hp], [a for a in apps if a["parent"] == hp])
                  for hp in sorted(set(a["parent"] for a in apps))]
    for name, train, tgt in groups:
        Q, th_all = [], []
        for bi in range(6):
            th = fit(kind, stack(train, bi, wfit), bi)
            th_all.append(th)
            Q.append(apply(kind, th, stack(tgt, bi, wfit)))
        params[name] = th_all
        alc, bud = u_per_app_alc(tgt, Q)
        for t, a in enumerate(tgt):
            i = idx_of[id(a)]
            d[i] = alc[t] - a["alc0"]
            db[i] = bud[t] - a["bud0"]
    return d, db, params


def u_summarise(apps, d, db, wkey, rng):
    """upd_probe.summarise, unrounded (the probe stored 5 decimals)."""
    out = {}
    for reg in REGS:
        idx = [i for i, a in enumerate(apps) if a["reg"] == reg]
        sub = [apps[i] for i in idx]
        w = np.array([a[wkey] for a in sub]); dd = d[idx]
        est = float((w * dd).sum() / w.sum() * (w.sum()))
        keys = [(a["parent"], a["sid"]) for a in sub]
        uk = sorted(set(keys)); kix = {k: t for t, k in enumerate(uk)}
        S = np.zeros(len(uk)); Wc = np.zeros(len(uk))
        for t in range(len(sub)):
            S[kix[keys[t]]] += w[t] * dd[t]; Wc[kix[keys[t]]] += w[t]
        bs = [S[p].sum() / Wc[p].sum() for p in (rng.integers(0, len(uk), len(uk)) for _ in range(1000))]
        pp = {}
        for par in sorted(set(a["parent"] for a in sub)):
            ii = [t for t, a in enumerate(sub) if a["parent"] == par]
            pp[par] = float((w[ii] * dd[ii]).sum() / w[ii].sum())
        bud = [float((w * db[idx][:, bi]).sum() / w.sum()) for bi in range(6)]
        bybin = {}
        for b in range(3):
            ii = [t for t, a in enumerate(sub) if a["bin"] == b]
            if ii:
                bybin[str(b)] = float((w[ii] * dd[ii]).sum() / w[ii].sum())
        out[reg] = dict(delta=est / w.sum(), se=float(np.std(bs)), per_parent=pp, brier_by_budget=bud,
                        by_vbin=bybin)
    return out


def u_baseline(apps):
    """hier-ship's excess over the pair-rate oracle by budget, raw and hidden-v weights."""
    res = {}
    for wkey in ("w_raw", "w_hv"):
        res[wkey] = {}
        for reg in REGS:
            sub = [a for a in apps if a["reg"] == reg]
            w = np.array([a[wkey] for a in sub])
            ex = np.array([a["bud0"] - a["v"] for a in sub])
            alc = np.array([a["alc0"] for a in sub]); v = np.array([a["v"] for a in sub])
            res[wkey][reg] = dict(alc=float((w * alc).sum() / w.sum()), v=float((w * v).sum() / w.sum()),
                                  excess_by_budget=[float((w * ex[:, bi]).sum() / w.sum()) for bi in range(6)],
                                  alc_minus_v=float((w * (alc - v)).sum() / w.sum()),
                                  b3_minus_b1=float((w * (ex[:, 2] - ex[:, 1])).sum() / w.sum()),
                                  b7_minus_b31=float((w * (ex[:, 3] - ex[:, 5])).sum() / w.sum()))
    return res


def _levers(apps, kinds, stack, fit, apply, rng, modes=("ins", "lopo", "lopo4")):
    out = {}
    for kind in kinds:
        for mode in modes:
            for wfit in ("w_raw", "w_hv"):
                d, db, params = u_run_mode(apps, kind, mode, wfit, stack, fit, apply)
                out[f"{kind}|{mode}|fit_{wfit}"] = dict(
                    params=params, score_raw=u_summarise(apps, d, db, "w_raw", rng),
                    score_hv=u_summarise(apps, d, db, "w_hv", rng))
    return out


def block_update_speed(base, vs, sh):
    apps = u_prepare(base, sh)
    out = dict(hidden_v=dict(n=len(vs), mean=float(vs.mean()), shares=sh.tolist(),
                             quantiles=np.quantile(vs, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()),
               replica_bin_shares={reg: (np.bincount([a["bin"] for a in apps if a["reg"] == reg], minlength=3) /
                                         sum(1 for a in apps if a["reg"] == reg)).tolist() for reg in REGS},
               baseline=u_baseline(apps))
    out["levers"] = _levers(apps, ("upd", "updp", "bba"), u_stack, u_fit, u_apply, np.random.default_rng(0))
    return out


# --- update_conditional: upd_probe2.py -------------------------------------------------------------

def u2_features(apps):
    by_run = {}
    for a in apps:
        by_run.setdefault((a["reg"], a["run"]), []).append(a)
    for grp in by_run.values():
        for a in grp:
            D = np.zeros(6)
            for bi, k in enumerate(BUD):
                if k == 0:
                    continue
                vals = []
                for b in grp:
                    if b is a:
                        continue
                    kk = min(k, len(b["y"]))
                    r = (b["y"][:kk].sum() + 1.0) / (kk + 2.0)
                    vals.append(abs(2 * r - 1))
                D[bi] = (np.mean(vals) - 0.5) if vals else 0.0
            a["D"] = D
            pat = np.zeros(6)
            for bi, k in enumerate(BUD):
                kk = min(k, len(a["y"]))
                if kk >= 2:
                    s = a["y"][:kk].sum()
                    pat[bi] = 1.0 if (s == 0 or s == kk) else 0.0
                else:
                    pat[bi] = 1.0
            a["pat"] = pat


def u2_stack(apps, bi, wkey):
    S = u_stack(apps, bi, wkey)
    S["pat"] = np.concatenate([np.full(len(a["K"]), a["pat"][bi]) for a in apps])
    S["D"] = np.concatenate([np.full(len(a["K"]), a["D"][bi]) for a in apps])
    pb = []
    for a in apps:
        pbar = float((a["P"][bi] * a["N"]).sum() / a["N"].sum())
        pb.append(np.full(len(a["K"]), pbar))
    S["pbar"] = np.concatenate(pb)
    return S


def u2_apply(kind, th, S):
    lp0, lp = logit(S["p0"]), logit(S["p"])
    if kind == "pat":
        lam = np.where(S["pat"] > 0.5, th[0], th[1])
        return sig(lp0 + lam * (lp - lp0))
    if kind == "spr":
        lb = logit(S["pbar"])
        return sig(lb + th[0] * (lp - lb))
    if kind == "disp":
        lam = np.exp(th[0] * S["D"])
        return sig(lp0 + lam * (lp - lp0))
    raise ValueError(kind)


def _u2_nm(kind, S, x0):
    r = minimize(lambda t: u_loss(u2_apply(kind, t, S), S), x0, method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-13, maxiter=2000))
    return [float(x) for x in r.x]


def u2_fit(kind, S, bi):
    if BUD[bi] == 0:
        return {"pat": [1.0, 1.0], "disp": [0.0]}[kind] if kind != "spr" else _u2_nm(kind, S, [1.0])
    x0 = {"pat": [1.0, 1.0], "spr": [1.0], "disp": [0.0]}[kind]
    return _u2_nm(kind, S, x0)


def block_update_conditional(base, sh):
    apps = u_prepare(base, sh)
    u2_features(apps)
    return {"levers": _levers(apps, ("pat", "spr", "disp"), u2_stack, u2_fit, u2_apply,
                              np.random.default_rng(0))}


# --- b0_stretch: b0_probe.py -----------------------------------------------------------------------

def b0_pack(apps, wkey):
    p0, pb, K, N, w = [], [], [], [], []
    for a in apps:
        m = len(a["K"])
        pbar = float((a["P"][0] * a["N"]).sum() / a["N"].sum())
        p0.append(a["P"][0]); pb.append(np.full(m, pbar)); K.append(a["K"]); N.append(a["N"])
        w.append(np.full(m, a[wkey] / a["N"].sum()))
    c = np.concatenate
    return dict(p0=c(p0), pb=c(pb), K=c(K), N=c(N), w=c(w))


def b0_q(kind, th, S, cfix):
    l0, lb = logit(S["p0"]), logit(S["pb"])
    if kind == "bw":
        return sig(cfix + th[0] * (lb - cfix) + (l0 - lb))
    if kind == "wi":
        return sig(lb + th[0] * (l0 - lb))
    if kind == "bwa":
        return sig(th[1] + th[0] * (lb - th[1]) + (l0 - lb))
    raise ValueError(kind)


def b0_fit(kind, S, cfix):
    x0 = [1.0] if kind != "bwa" else [1.0, cfix]
    r = minimize(lambda t: u_loss(b0_q(kind, t, S, cfix), S), x0, method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-13, maxiter=3000))
    return [float(x) for x in r.x]


def b0_per_app(apps, q):
    out, off = [], 0
    for a in apps:
        m = len(a["K"]); qq = q[off:off + m]; off += m
        out.append(float((a["K"] * (1 - qq) ** 2 + (a["N"] - a["K"]) * qq ** 2).sum() / a["N"].sum()))
    return np.array(out)


def block_b0_stretch(base, sh):
    apps = u_prepare(base, sh)
    for a in apps:
        a["b00"] = float((a["K"] * (1 - a["P"][0]) ** 2 + (a["N"] - a["K"]) * a["P"][0] ** 2).sum()
                         / a["N"].sum())
    desc = {}
    for reg in REGS:
        sub = [a for a in apps if a["reg"] == reg]
        S = b0_pack(sub, "w_raw")
        l0, lb = logit(S["p0"]), logit(S["pb"])
        wN = S["w"] * S["N"]
        tot = np.average((l0 - np.average(l0, weights=wN)) ** 2, weights=wN)
        btw = np.average((lb - np.average(lb, weights=wN)) ** 2, weights=wN)
        cfix = float(np.average(lb, weights=wN))
        row = dict(var_total=float(tot), var_between=float(btw), centre=cfix,
                   b_bw=b0_fit("bw", S, cfix)[0], k_wi=b0_fit("wi", S, cfix)[0], bwa=b0_fit("bwa", S, cfix))
        row["b_bw_by_parent"] = {par: b0_fit("bw", b0_pack([a for a in sub if a["parent"] == par], "w_raw"),
                                             cfix)[0]
                                 for par in sorted(set(a["parent"] for a in sub))}
        desc[reg] = row
    rng = np.random.default_rng(0)
    res = {}
    for kind in ("bw", "wi", "bwa"):
        for mode in ("lopo", "lopo4"):
            for wfit in ("w_raw", "w_hv"):
                d = np.zeros(len(apps)); params = {}
                pool = [a for a in apps if (a["reg"] in FIT_ON if mode == "lopo" else True)]
                for hp in sorted(set(a["parent"] for a in apps)):
                    train = [a for a in pool if a["parent"] != hp]
                    tgt_i = [i for i, a in enumerate(apps) if a["parent"] == hp]
                    tgt = [apps[i] for i in tgt_i]
                    Str = b0_pack(train, wfit)
                    cfix = float(np.average(logit(Str["pb"]), weights=Str["w"] * Str["N"]))
                    th = b0_fit(kind, Str, cfix)
                    params[hp] = dict(th=th, c=cfix)
                    b0 = b0_per_app(tgt, b0_q(kind, th, b0_pack(tgt, wfit), cfix))
                    for t, i in enumerate(tgt_i):
                        d[i] = 0.1 * (b0[t] - apps[i]["b00"])
                db = np.zeros((len(apps), 6)); db[:, 0] = d / 0.1
                res[f"{kind}|{mode}|fit_{wfit}"] = dict(params=params,
                                                         score_raw=u_summarise(apps, d, db, "w_raw", rng),
                                                         score_hv=u_summarise(apps, d, db, "w_hv", rng))
    return {"describe": desc, "levers": res}


# --- damping: info/damp_probe.py --------------------------------------------------------------------

def block_damping(base, vs):
    apps = fresh(base)
    for a in apps:
        a["R"] = np.zeros(6)
        r = a["K"].sum() / a["N"].sum()
        a["v"] = r * (1 - r)
    h = np.histogram(np.clip(vs, 0, 0.25), DAMP_BINS)[0].astype(float)
    hshare = h / h.sum()
    wmap = {}
    for reg in REGS:
        sub = [a for a in apps if a["reg"] == reg]
        b = np.digitize([min(a["v"], 0.25) for a in sub], DAMP_BINS[1:-1])
        share = np.bincount(b, minlength=3) / len(sub)
        for a, bb in zip(sub, b):
            a["vw"] = hshare[bb] / share[bb]
        wmap[reg] = dict(replica_share=share.tolist())
    tr_pool = [a for a in apps if a["reg"] in FIT_ON]
    held = sorted(set(a["parent"] for a in apps))
    res = {}
    for mode in ("plain", "vweighted_fit"):
        diff = np.zeros((len(apps), 6))
        lams = {}
        for hp in held:
            train = [a for a in tr_pool if a["parent"] != hp]
            idx = [i for i, a in enumerate(apps) if a["parent"] == hp]
            lams[hp] = []
            for bi in range(6):
                if bi == 0:
                    lams[hp].append(1.0)
                    continue
                S = pp_stack(train, bi)
                S0 = pp_stack(train, 0)
                if mode == "vweighted_fit":
                    S["w"] = S["w"] * np.concatenate([np.full(len(a["K"]), a["vw"]) for a in train])
                l0, lk = logit(S0["p"]), logit(S["p"])
                lam = float(minimize_scalar(lambda t: pp_loss(sig(l0 + t * (lk - l0)), S), bounds=(0.3, 1.6),
                                            method="bounded").x)
                lams[hp].append(lam)
                for i in idx:
                    a = apps[i]
                    q = sig(logit(a["P"][0]) + lam * (logit(a["P"][bi]) - logit(a["P"][0])))
                    diff[i, bi] = brier(a["K"], a["N"], q) - brier(a["K"], a["N"], a["P"][bi])
        out = {"lambda": lams}
        for reg in REGS:
            for wt in ("equal", "hidden_v"):
                ii = [i for i, a in enumerate(apps) if a["reg"] == reg]
                w = np.array([1.0 / apps[i]["n"] * (apps[i]["vw"] if wt == "hidden_v" else 1.0) for i in ii])
                d = diff[ii] @ W6
                est = float((w * d).sum() / w.sum() * np.mean([1.0]))
                keys = [(apps[i]["parent"], apps[i]["sid"]) for i in ii]
                uk = {k: t for t, k in enumerate(sorted(set(keys)))}
                Sx = np.zeros(len(uk)); Wx = np.zeros(len(uk))
                for t, k in enumerate(keys):
                    Sx[uk[k]] += w[t] * d[t]; Wx[uk[k]] += w[t]
                bs = _bootstrap(Sx, Wx, np.random.default_rng(1), 1000)
                bud = [float((w * diff[ii, bi]).sum() / w.sum()) for bi in range(6)]
                out[f"{reg}_{wt}"] = dict(delta_per_pair=est, se=float(np.std(bs)), by_budget=bud)
        res[mode] = out
    res["hidden_v"] = dict(n=len(vs), mean=float(vs.mean()), shares=hshare.tolist(), bins=DAMP_BINS)
    res["replica_shares"] = wmap
    return res


# --- budget_switch: budget_switch_probe.py ----------------------------------------------------------

def block_budget_switch(rs_dir=RS_ROWS):
    out, M = {}, {}
    for reg in RS_REGIMES:
        runs = []
        for f in sorted(glob.glob(os.path.join(rs_dir, reg, "*.json"))):
            with open(f) as fh:
                runs.append(json.load(fh))
        if not runs:
            return None
        cfgs = list(RS_CONFIGS)
        mb = {}
        for c in cfgs:
            per_run = []
            for d in runs:
                b = np.array(d["res"][c]["b"], float)
                per_run.append(b.mean(axis=0) if b.ndim == 2 else b)
            mb[c] = np.mean(per_run, axis=0)
        M[reg] = mb
        best = [min(cfgs, key=lambda c: mb[c][k]) for k in range(6)]
        out[reg] = dict(n_runs=len(runs), budget_diff_vs_ship={c: (mb[c] - mb["ship"]).tolist() for c in cfgs},
                        best_per_budget=best,
                        in_sample_switch_gain=float(sum(W6[k] * (mb[best[k]][k] - mb["ship"][k]) for k in range(6))),
                        alc={c: float(W6 @ mb[c]) for c in cfgs},
                        configs_in_rows=sorted({c for d in runs for c in d["res"]}))
    cfgs = list(RS_CONFIGS)
    hier_like = [c for c in cfgs if c not in ("smooth", "smcal", "legacy")]

    def worst(s):
        return max(sum(W6[j] * (M[r][s[j]][j] - M[r]["ship"][j]) for j in range(6)) for r in RS_REGIMES)
    for name, cand_set in (("all", cfgs), ("hier_only", hier_like)):
        sched, improved = ["ship"] * 6, True
        while improved:
            improved = False
            for k in range(6):
                for c in cand_set:
                    s2 = list(sched); s2[k] = c
                    if worst(s2) < worst(sched) - 1e-9:
                        sched, improved = s2, True
        per_reg = {r: float(sum(W6[j] * (M[r][sched[j]][j] - M[r]["ship"][j]) for j in range(6)))
                   for r in RS_REGIMES}
        out["robust_switch_" + name] = dict(schedule=sched, per_regime=per_reg)
    return out


# --- the family table ------------------------------------------------------------------------------

#: lever -> (label, block, key of its LOPO-on-TL+mix reading, score key, delta key, se key)
FAMILY = {
    "tonly": ("per-budget temperature", "maps", None),
    "temp": ("per-budget affine logit map", "maps", None),
    "iso": ("per-budget isotonic map", "maps", None),
    "blend": ("blend with Smooth-cal", "maps", None),
    "blend22": ("blend with Smooth, Beta(2,2)", "maps", None),
    "runshift": ("run-level shift from other pairs' labels", "maps", None),
    "upd": ("update damping lambda_k (logit)", "update_speed", "upd|lopo|fit_w_raw"),
    "updp": ("update damping lambda_k (probability)", "update_speed", "updp|lopo|fit_w_raw"),
    "bba": ("blend with a Beta-binomial learner on hier's B0 mean", "update_speed", "bba|lopo|fit_w_raw"),
    "pat": ("lambda by the pair's own label pattern", "update_conditional", "pat|lopo|fit_w_raw"),
    "spr": ("within-pair spread kappa_k", "update_conditional", "spr|lopo|fit_w_raw"),
    "disp": ("lambda by the run's label dispersion", "update_conditional", "disp|lopo|fit_w_raw"),
    "bw": ("B0 between-pair stretch", "b0_stretch", "bw|lopo|fit_w_raw"),
    "wi": ("B0 within-pair stretch", "b0_stretch", "wi|lopo|fit_w_raw"),
    "bwa": ("B0 stretch with its centre fitted", "b0_stretch", "bwa|lopo|fit_w_raw"),
    "damp": ("damping lambda_k in [0.3, 1.6], run weights", "damping", "plain"),
}


def _reading(res, lever):
    """{reg: (delta, se)} of the lever's primary reading: LOPO on TL+mix, raw weights."""
    label, block, key = FAMILY[lever]
    if block == "maps":
        s = res["maps"]["summary"][lever]
        return {r: (s[r]["delta"], s[r]["cluster_se"]) for r in REGS}
    if block == "damping":
        s = res["damping"][key]
        return {r: (s[f"{r}_equal"]["delta_per_pair"], s[f"{r}_equal"]["se"]) for r in REGS}
    s = res[block]["levers"][key]["score_raw"]
    return {r: (s[r]["delta"], s[r]["se"]) for r in REGS}


def _verdict(rd):
    tl, se = rd["tl"]
    worst_public = max(rd["r1b"][0], rd["r1p"][0])
    if tl <= GATE_TL and worst_public <= GUARD_PUBLIC:
        return "passes the test-like bar and the public guard"
    if (tl > 2 * se and tl >= NEG_MIN) or worst_public > GUARD_PUBLIC:
        return "negative"
    return "null"


def family(res):
    rows = {}
    for lever, (label, block, key) in FAMILY.items():
        rd = _reading(res, lever)
        rows[lever] = {"label": label, "block": block, "key": key or lever,
                       "delta": {r: rd[r][0] for r in REGS}, "se": {r: rd[r][1] for r in REGS},
                       "verdict": _verdict(rd)}
    # every LOPO-fitted variant, raw scoring: the best test-like delta
    cands = []
    for lever in PP_KINDS:
        s = res["maps"]["summary"][lever]["tl"]
        cands.append((s["delta"], s["cluster_se"], f"maps.{lever}"))
    for block in ("update_speed", "update_conditional", "b0_stretch"):
        for key, v in res[block]["levers"].items():
            if "|ins|" in key:
                continue
            s = v["score_raw"]["tl"]
            cands.append((s["delta"], s["se"], f"{block}.{key}"))
    for mode in ("plain", "vweighted_fit"):
        s = res["damping"][mode]["tl_equal"]
        cands.append((s["delta_per_pair"], s["se"], f"damping.{mode}"))
    cands.sort()
    best = cands[0]
    bs = res.get("budget_switch")
    out = {"rows": rows,
           "best_tl_lopo": {"delta": best[0], "se": best[1], "variant": best[2]},
           "lopo_variants": len(cands),
           "lopo_variants_with_tl_delta_below_0": [{"variant": c[2], "delta": c[0], "se": c[1]}
                                                   for c in cands if c[0] < 0],
           "any_meets_tl_bar": any(c[0] <= GATE_TL for c in cands),
           "verdicts": sorted({r["verdict"] for r in rows.values()}),
           "budget_switch": None if bs is None else {
               "robust_schedule": bs["robust_switch_all"]["schedule"],
               "robust_worst_regime_gain": max(bs["robust_switch_all"]["per_regime"].values()),
               "robust_best_regime_gain": min(bs["robust_switch_all"]["per_regime"].values()),
               "in_sample_gain_range": [min(bs[r]["in_sample_switch_gain"] for r in RS_REGIMES),
                                        max(bs[r]["in_sample_switch_gain"] for r in RS_REGIMES)]},
           "parameter_spread": spread(res)}
    return out


def spread(res):
    """'The optimum belongs to the parent': the fitted parameter across held-out parents."""
    m = res["maps"]["summary"]
    upd = res["update_speed"]["levers"]["upd|lopo|fit_w_raw"]["params"]
    lam1 = {hp: v[1][0] for hp, v in upd.items()}
    lam1_others = [v for hp, v in lam1.items() if hp != "multi_swebench"]
    by_parent = [v for reg in REGS for v in res["b0_stretch"]["describe"][reg]["b_bw_by_parent"].values()]
    return {
        "tonly_b0_temperature_by_fold": {hp: v[0][0] for hp, v in m["tonly"]["params"].items()},
        "tonly_b0_temperature_range": [min(v[0][0] for v in m["tonly"]["params"].values()),
                                       max(v[0][0] for v in m["tonly"]["params"].values())],
        "blend_b0_weight_by_fold": {hp: v[0][0] for hp, v in m["blend"]["params"].items()},
        "blend_b0_weight_range": [min(v[0][0] for v in m["blend"]["params"].values()),
                                  max(v[0][0] for v in m["blend"]["params"].values())],
        "runshift_c_by_fold": {hp: [x[0] for x in v] for hp, v in m["runshift"]["params"].items()},
        "runshift_c_median_over_folds": [float(np.median([v[bi][0] for v in m["runshift"]["params"].values()]))
                                         for bi in range(6)],
        "runshift_c_range_over_folds": [[min(v[bi][0] for v in m["runshift"]["params"].values()),
                                         max(v[bi][0] for v in m["runshift"]["params"].values())]
                                        for bi in range(6)],
        "upd_lambda1_by_fold": lam1,
        "upd_lambda1_without_multi_swebench": lam1["multi_swebench"],
        "upd_lambda1_range_other_folds": [min(lam1_others), max(lam1_others)],
        "b0_between_pair_slope_by_parent_range": [min(by_parent), max(by_parent)],
        "calibration_b0_slope_range": [min(res["maps"]["calibration"][r][0]["b"] for r in REGS),
                                       max(res["maps"]["calibration"][r][0]["b"] for r in REGS)],
        "affine_in_sample_ceiling_alc": {r: float(sum(W6[bi] * res["maps"]["calibration"][r][bi]["gain_in_sample"]
                                                      for bi in range(6))) for r in REGS},
        "calibration_b0_mean_p_minus_mean_y": {r: res["maps"]["calibration"][r][0]["mean_p"] -
                                               res["maps"]["calibration"][r][0]["mean_y"] for r in REGS},
    }


# --- the scratch values ----------------------------------------------------------------------------

#: The probes' printed tables (stdout, 5 decimals), verbatim: checked on every run, with or
#: without the scratch directory. posthoc_probe.out (its first six lines), upd_probe.out,
#: upd_probe2.out and b0_probe.out (their lever lines).
PRINTED = {
    "maps": """\
tonly | tl +0.00020 (0.00069) | mix +0.00206 (0.00070) | r1b +0.01091 (0.00115) | r1p +0.01165 (0.00099)
temp | tl +0.00133 (0.00161) | mix +0.00307 (0.00129) | r1b +0.01503 (0.00192) | r1p +0.01732 (0.00195)
blend | tl +0.00157 (0.00110) | mix +0.00336 (0.00099) | r1b +0.00507 (0.00087) | r1p +0.00763 (0.00105)
blend22 | tl +0.00000 (0.00000) | mix +0.00001 (0.00000) | r1b +0.00000 (0.00000) | r1p +0.00001 (0.00000)
runshift | tl -0.00034 (0.00018) | mix +0.00005 (0.00025) | r1b +0.00003 (0.00048) | r1p +0.00042 (0.00039)
iso | tl +0.00239 (0.00165) | mix +0.00290 (0.00113) | r1b +0.01239 (0.00167) | r1p +0.01514 (0.00184)""",
    "update_speed": """\
upd|ins|fit_w_raw tl:-0.00002(0.00003)/hv+0.00005 mix:-0.00007(0.00011)/hv-0.00018 r1b:-0.00106(0.00059)/hv-0.00024 r1p:-0.00060(0.00032)/hv-0.00018
upd|ins|fit_w_hv tl:+0.00034(0.00027)/hv-0.00031 mix:+0.00008(0.00034)/hv-0.00033 r1b:-0.00082(0.00034)/hv-0.00048 r1p:-0.00052(0.00021)/hv-0.00026
upd|lopo|fit_w_raw tl:+0.00091(0.00027)/hv+0.00033 mix:+0.00149(0.00040)/hv+0.00078 r1b:+0.00002(0.00016)/hv+0.00013 r1p:-0.00033(0.00019)/hv-0.00025
upd|lopo|fit_w_hv tl:+0.00095(0.00039)/hv+0.00001 mix:+0.00113(0.00052)/hv+0.00033 r1b:-0.00060(0.00036)/hv-0.00003 r1p:-0.00053(0.00032)/hv-0.00010
upd|lopo4|fit_w_raw tl:+0.00065(0.00027)/hv-0.00007 mix:+0.00055(0.00033)/hv+0.00009 r1b:-0.00018(0.00036)/hv+0.00051 r1p:-0.00007(0.00038)/hv+0.00049
upd|lopo4|fit_w_hv tl:+0.00065(0.00029)/hv-0.00002 mix:+0.00032(0.00030)/hv+0.00002 r1b:+0.00025(0.00047)/hv+0.00113 r1p:+0.00060(0.00049)/hv+0.00136
updp|ins|fit_w_raw tl:-0.00003(0.00006)/hv+0.00013 mix:-0.00005(0.00006)/hv-0.00011 r1b:-0.00115(0.00063)/hv-0.00032 r1p:-0.00066(0.00034)/hv-0.00022
updp|ins|fit_w_hv tl:+0.00039(0.00026)/hv-0.00027 mix:+0.00016(0.00034)/hv-0.00030 r1b:-0.00094(0.00036)/hv-0.00054 r1p:-0.00058(0.00022)/hv-0.00031
updp|lopo|fit_w_raw tl:+0.00102(0.00030)/hv+0.00044 mix:+0.00167(0.00039)/hv+0.00091 r1b:+0.00011(0.00017)/hv+0.00015 r1p:-0.00024(0.00018)/hv-0.00020
updp|lopo|fit_w_hv tl:+0.00113(0.00039)/hv+0.00009 mix:+0.00137(0.00054)/hv+0.00043 r1b:-0.00064(0.00029)/hv-0.00016 r1p:-0.00061(0.00029)/hv-0.00025
updp|lopo4|fit_w_raw tl:+0.00080(0.00031)/hv-0.00001 mix:+0.00083(0.00038)/hv+0.00018 r1b:-0.00035(0.00029)/hv+0.00018 r1p:-0.00031(0.00030)/hv+0.00012
updp|lopo4|fit_w_hv tl:+0.00078(0.00031)/hv-0.00001 mix:+0.00049(0.00033)/hv+0.00003 r1b:-0.00007(0.00042)/hv+0.00068 r1p:+0.00019(0.00043)/hv+0.00084
bba|ins|fit_w_raw tl:-0.00000(0.00001)/hv+0.00000 mix:-0.00001(0.00005)/hv-0.00007 r1b:-0.00107(0.00055)/hv-0.00033 r1p:-0.00050(0.00029)/hv-0.00017
bba|ins|fit_w_hv tl:+0.00036(0.00024)/hv-0.00024 mix:+0.00018(0.00027)/hv-0.00022 r1b:-0.00091(0.00035)/hv-0.00050 r1p:-0.00044(0.00019)/hv-0.00023
bba|lopo|fit_w_raw tl:+0.00088(0.00027)/hv+0.00025 mix:+0.00135(0.00040)/hv+0.00061 r1b:-0.00024(0.00013)/hv-0.00016 r1p:-0.00043(0.00017)/hv-0.00036
bba|lopo|fit_w_hv tl:+0.00109(0.00037)/hv+0.00014 mix:+0.00142(0.00050)/hv+0.00053 r1b:-0.00053(0.00027)/hv-0.00013 r1p:-0.00045(0.00027)/hv-0.00014
bba|lopo4|fit_w_raw tl:+0.00074(0.00025)/hv+0.00004 mix:+0.00084(0.00035)/hv+0.00027 r1b:-0.00022(0.00026)/hv+0.00022 r1p:-0.00013(0.00030)/hv+0.00025
bba|lopo4|fit_w_hv tl:+0.00073(0.00027)/hv+0.00004 mix:+0.00057(0.00031)/hv+0.00015 r1b:+0.00005(0.00040)/hv+0.00070 r1p:+0.00042(0.00043)/hv+0.00101""",
    "update_conditional": """\
pat|ins|fit_w_raw tl:-0.00007(0.00005)/hv+0.00007 mix:-0.00009(0.00011)/hv-0.00023 r1b:-0.00120(0.00063)/hv-0.00025 r1p:-0.00076(0.00034)/hv-0.00039
pat|ins|fit_w_hv tl:+0.00035(0.00027)/hv-0.00033 mix:+0.00015(0.00035)/hv-0.00044 r1b:-0.00091(0.00035)/hv-0.00054 r1p:-0.00065(0.00025)/hv-0.00049
pat|lopo|fit_w_raw tl:+0.00092(0.00027)/hv+0.00038 mix:+0.00155(0.00040)/hv+0.00089 r1b:+0.00003(0.00016)/hv+0.00013 r1p:-0.00028(0.00019)/hv-0.00018
pat|lopo|fit_w_hv tl:+0.00110(0.00040)/hv+0.00004 mix:+0.00121(0.00053)/hv+0.00030 r1b:-0.00063(0.00034)/hv-0.00010 r1p:-0.00060(0.00032)/hv-0.00027
pat|lopo4|fit_w_raw tl:+0.00077(0.00029)/hv-0.00004 mix:+0.00072(0.00036)/hv+0.00017 r1b:-0.00012(0.00039)/hv+0.00064 r1p:+0.00017(0.00039)/hv+0.00081
pat|lopo4|fit_w_hv tl:+0.00087(0.00032)/hv-0.00002 mix:+0.00052(0.00034)/hv+0.00002 r1b:+0.00026(0.00048)/hv+0.00114 r1p:+0.00075(0.00051)/hv+0.00152
spr|ins|fit_w_raw tl:-0.00000(0.00001)/hv-0.00000 mix:-0.00039(0.00018)/hv-0.00032 r1b:-0.00015(0.00009)/hv+0.00001 r1p:-0.00032(0.00014)/hv-0.00018
spr|ins|fit_w_hv tl:-0.00000(0.00001)/hv-0.00001 mix:-0.00038(0.00016)/hv-0.00033 r1b:-0.00010(0.00004)/hv-0.00003 r1p:-0.00031(0.00012)/hv-0.00019
spr|lopo|fit_w_raw tl:+0.00018(0.00006)/hv+0.00018 mix:-0.00000(0.00004)/hv+0.00001 r1b:+0.00000(0.00007)/hv+0.00007 r1p:+0.00003(0.00004)/hv+0.00006
spr|lopo|fit_w_hv tl:+0.00017(0.00006)/hv+0.00017 mix:-0.00001(0.00004)/hv+0.00000 r1b:-0.00001(0.00007)/hv+0.00006 r1p:+0.00001(0.00004)/hv+0.00005
spr|lopo4|fit_w_raw tl:+0.00021(0.00007)/hv+0.00021 mix:-0.00007(0.00005)/hv-0.00005 r1b:+0.00005(0.00009)/hv+0.00015 r1p:-0.00003(0.00005)/hv+0.00003
spr|lopo4|fit_w_hv tl:+0.00015(0.00006)/hv+0.00014 mix:-0.00005(0.00004)/hv-0.00004 r1b:+0.00003(0.00006)/hv+0.00011 r1p:-0.00001(0.00004)/hv+0.00003
disp|ins|fit_w_raw tl:-0.00001(0.00001)/hv-0.00001 mix:-0.00004(0.00009)/hv-0.00014 r1b:-0.00086(0.00045)/hv-0.00026 r1p:-0.00034(0.00022)/hv-0.00011
disp|ins|fit_w_hv tl:+0.00022(0.00022)/hv-0.00023 mix:+0.00007(0.00026)/hv-0.00025 r1b:-0.00073(0.00029)/hv-0.00039 r1p:-0.00030(0.00015)/hv-0.00016
disp|lopo|fit_w_raw tl:+0.00072(0.00021)/hv+0.00027 mix:+0.00107(0.00032)/hv+0.00053 r1b:-0.00000(0.00012)/hv+0.00009 r1p:-0.00023(0.00015)/hv-0.00012
disp|lopo|fit_w_hv tl:+0.00069(0.00028)/hv+0.00003 mix:+0.00081(0.00038)/hv+0.00022 r1b:-0.00034(0.00023)/hv-0.00008 r1p:-0.00020(0.00024)/hv+0.00006
disp|lopo4|fit_w_raw tl:+0.00046(0.00019)/hv+0.00001 mix:+0.00043(0.00026)/hv+0.00013 r1b:-0.00008(0.00030)/hv+0.00043 r1p:+0.00024(0.00031)/hv+0.00071
disp|lopo4|fit_w_hv tl:+0.00047(0.00022)/hv-0.00000 mix:+0.00030(0.00024)/hv+0.00008 r1b:+0.00029(0.00036)/hv+0.00088 r1p:+0.00080(0.00040)/hv+0.00140""",
    "b0_stretch": """\
bw|lopo|fit_w_raw tl:+0.00004(0.00019)/hv+0.00022 mix:+0.00044(0.00023)/hv+0.00048 r1b:+0.00260(0.00037)/hv+0.00244 r1p:+0.00312(0.00042)/hv+0.00298
bw|lopo|fit_w_hv tl:+0.00004(0.00017)/hv+0.00020 mix:+0.00037(0.00021)/hv+0.00044 r1b:+0.00167(0.00027)/hv+0.00148 r1p:+0.00194(0.00031)/hv+0.00179
bw|lopo4|fit_w_raw tl:-0.00006(0.00021)/hv+0.00003 mix:+0.00012(0.00021)/hv+0.00037 r1b:+0.00004(0.00029)/hv+0.00028 r1p:-0.00025(0.00016)/hv-0.00009
bw|lopo4|fit_w_hv tl:+0.00005(0.00014)/hv+0.00007 mix:+0.00026(0.00012)/hv+0.00039 r1b:-0.00007(0.00018)/hv+0.00008 r1p:-0.00025(0.00010)/hv-0.00015
wi|lopo|fit_w_raw tl:+0.00000(0.00000)/hv+0.00000 mix:+0.00000(0.00000)/hv+0.00000 r1b:-0.00000(0.00000)/hv-0.00000 r1p:-0.00000(0.00000)/hv-0.00000
wi|lopo|fit_w_hv tl:+0.00000(0.00000)/hv+0.00000 mix:+0.00000(0.00000)/hv+0.00000 r1b:-0.00000(0.00000)/hv-0.00000 r1p:-0.00000(0.00000)/hv-0.00000
wi|lopo4|fit_w_raw tl:+0.00000(0.00000)/hv+0.00000 mix:+0.00000(0.00000)/hv+0.00000 r1b:-0.00000(0.00000)/hv-0.00000 r1p:-0.00000(0.00000)/hv-0.00000
wi|lopo4|fit_w_hv tl:+0.00000(0.00000)/hv+0.00000 mix:+0.00000(0.00000)/hv+0.00000 r1b:-0.00000(0.00000)/hv-0.00000 r1p:-0.00000(0.00000)/hv-0.00000
bwa|lopo|fit_w_raw tl:+0.00050(0.00058)/hv+0.00106 mix:+0.00165(0.00053)/hv+0.00218 r1b:+0.00861(0.00087)/hv+0.00865 r1p:+0.00983(0.00090)/hv+0.01002
bwa|lopo|fit_w_hv tl:+0.00029(0.00045)/hv+0.00073 mix:+0.00123(0.00042)/hv+0.00165 r1b:+0.00629(0.00068)/hv+0.00638 r1p:+0.00753(0.00071)/hv+0.00770
bwa|lopo4|fit_w_raw tl:+0.00258(0.00051)/hv+0.00225 mix:+0.00369(0.00036)/hv+0.00344 r1b:+0.00203(0.00036)/hv+0.00273 r1p:+0.00316(0.00029)/hv+0.00371
bwa|lopo4|fit_w_hv tl:+0.00311(0.00052)/hv+0.00258 mix:+0.00428(0.00043)/hv+0.00382 r1b:+0.00176(0.00033)/hv+0.00243 r1p:+0.00297(0.00025)/hv+0.00347""",
}

#: damp_probe.json and budget_switch_probe.json, which have no printed table: the values the
#: ledger and critic.md quote (5 decimals), from the stored JSON.
PRINTED_DAMPING = {  # mode/cell: (delta_per_pair, se)
    "plain/tl_equal": (0.00071, 0.00022), "plain/tl_hidden_v": (0.00027, 0.00017),
    "plain/mix_equal": (0.00117, 0.00030), "plain/mix_hidden_v": (0.00062, 0.00024),
    "plain/r1b_equal": (0.00011, 0.00014), "plain/r1b_hidden_v": (0.00018, 0.00014),
    "vweighted_fit/tl_equal": (0.00084, 0.00036), "vweighted_fit/tl_hidden_v": (-0.00006, 0.00028),
    "vweighted_fit/mix_equal": (0.00093, 0.00046), "vweighted_fit/mix_hidden_v": (0.00021, 0.00034),
    "vweighted_fit/r1b_equal": (-0.00062, 0.00035), "vweighted_fit/r1b_hidden_v": (-0.00004, 0.00041),
}
PRINTED_SWITCH = {  # regime: in-sample switch gain, and the robust schedule's per-regime gain (5 dp)
    "in_sample": {"TUNED": -0.00382, "READING": -0.00003, "AUDIT": -0.00074, "MIXTURE": -0.00032,
                  "FLAT": -0.00101, "R1B": -0.0023, "R1P": -0.00035},
    "robust_schedule": ["ship", "ship", "ship", "ship", "ship", "aggr"],
    "robust_worst": -0.0,
}

#: statements of the ledger and critic.md about this family, at their stated precision
#: (path into the results, decimals, value, source)
STATEMENTS = [
    (("maps", "summary", "temp", "r1b", "delta"), 5, 0.01503, "ledger 2G: affine R1-bf +0.01503"),
    (("maps", "summary", "runshift", "tl", "delta"), 5, -0.00034, "ledger 2G and outline T4 note: -0.00034"),
    (("maps", "summary", "runshift", "tl", "cluster_se"), 5, 0.00018, "same: (0.00018)"),
    (("maps", "summary", "temp", "tl", "base_alc"), 5, 0.16551, "ledger 2I: base ALC TL 0.16551"),
    (("maps", "summary", "temp", "mix", "base_alc"), 5, 0.17197, "ledger: mix 0.17197"),
    (("maps", "summary", "temp", "r1b", "base_alc"), 5, 0.20499, "ledger: R1-bf 0.20499"),
    (("maps", "summary", "temp", "r1p", "base_alc"), 5, 0.20153, "ledger: R1-pu 0.20153"),
    (("family", "parameter_spread", "affine_in_sample_ceiling_alc", "tl"), 4, -0.0043,
     "ledger 2G: in-sample ceiling of per-budget affine calibration -0.0043 TL"),
    (("family", "parameter_spread", "affine_in_sample_ceiling_alc", "mix"), 4, -0.0044, "same: -0.0044 mix"),
    (("family", "parameter_spread", "affine_in_sample_ceiling_alc", "r1b"), 4, -0.0031, "same: -0.0031 R1-bf"),
    (("family", "parameter_spread", "affine_in_sample_ceiling_alc", "r1p"), 4, -0.0030, "same: -0.0030 R1-pu"),
    (("maps", "calibration", "tl", 0, "b"), 2, 3.19, "ledger 2G: the in-sample B0 slope b about 2.5 to 3.2"),
    (("maps", "calibration", "mix", 0, "b"), 2, 2.53, "same (posthoc_probe.out calibration line)"),
    (("maps", "calibration", "tl", 0, "mean_p"), 3, 0.416, "ledger 2G: TL mean p 0.416 vs y 0.280"),
    (("maps", "calibration", "tl", 0, "mean_y"), 2, 0.28, "same"),
    (("maps", "calibration", "r1b", 0, "mean_p"), 3, 0.302, "ledger 2G: R1 0.302 vs 0.385"),
    (("maps", "calibration", "r1b", 0, "mean_y"), 3, 0.385, "same"),
    (("family", "parameter_spread", "tonly_b0_temperature_range", 0), 2, 1.57,
     "ledger 3.11: the B0 temperature differs across folds from 1.57 to 3.8"),
    (("family", "parameter_spread", "tonly_b0_temperature_range", 1), 1, 3.8, "same"),
    (("family", "parameter_spread", "blend_b0_weight_by_fold", "matharena"), 1, 1.0,
     "ledger 2G: w 1.0 at B0 with matharena out"),
    (("family", "parameter_spread", "blend_b0_weight_by_fold", "multi_swebench"), 2, 0.09,
     "ledger 2G: about 0.09 with multi_swebench out"),
    (("family", "parameter_spread", "upd_lambda1_without_multi_swebench"), 2, 0.65,
     "critic G: lambda at B1 is 0.65 without multi_swebench"),
    (("family", "parameter_spread", "upd_lambda1_range_other_folds", 0), 2, 0.95,
     "critic G: 0.95 to 1.07 without each other parent"),
    (("family", "parameter_spread", "upd_lambda1_range_other_folds", 1), 2, 1.07, "same"),
    (("family", "parameter_spread", "b0_between_pair_slope_by_parent_range", 0), 1, 0.5,
     "critic G: the B0 slope ranges from 0.5 to 20 by parent"),
    (("family", "parameter_spread", "b0_between_pair_slope_by_parent_range", 1), 0, 20, "same"),
    (("family", "parameter_spread", "runshift_c_median_over_folds", 1), 2, 0.36,
     "ledger 2G: c_k about 0.36 at B1/B3, about 0.07 at B31"),
    (("family", "parameter_spread", "runshift_c_median_over_folds", 2), 2, 0.36, "same"),
    (("family", "parameter_spread", "runshift_c_median_over_folds", 5), 2, 0.07, "same"),
]

#: scratch values the port does not reproduce as stated, and why. Every number the probes stored
#: or printed reproduces; these are roundings in the ledger's and critic.md's prose.
EXPLAINED = {
    "family/parameter_spread/upd_lambda1_range_other_folds/1":
        "critic.md (section G) wrote 'lambda at B1 is 0.65 without multi_swebench and 0.95 to 1.07 without "
        "each other parent'. The largest of the four other folds is 1.0647 (researchcodebench held out), "
        "1.06 at two decimals; upd_probe.json stores it unrounded, and 1.07 rounds its three-decimal print "
        "(1.065) a second time. The reading is unchanged: 0.65 against 0.95 to 1.06.",
    "family/parameter_spread/runshift_c_median_over_folds/2":
        "the ledger (section 2G) gives c_k 'about 0.36 at B1/B3'. At B3 the five folds give 0.105 "
        "(multi_swebench held out), 0.295, 0.315, 0.362 and 0.364 (matharena and real_webagents held "
        "out): 0.36 is the value of two folds, the median is 0.315. At B1 the median is 0.357 (0.36).",
    "family/parameter_spread/runshift_c_median_over_folds/5":
        "the ledger's 'about 0.07 at B31': the five folds give 0.067 to 0.083, median 0.0751 (0.08 at two "
        "decimals).",
}

SCRATCH_FILES = {  # block: (scratch path relative to the scratch dir, sha256 of the stored JSON)
    "maps": ("posthoc_probe.json", "a2b0b352be2865b52f1418d550bc39365315cb3826cdc737696134189c4f5935"),
    "update_speed": ("upd_probe.json", "9b16758a16dd46dd5839419efde9d17f6a102634a7120500c0e9b1deaac849ca"),
    "update_conditional": ("upd_probe2.json", "b8954ac69c6cdaa79b3929690d7f09e93a38d9d327f5daf0ade72cd8618750fb"),
    "b0_stretch": ("b0_probe.json", "cad0db3269603abcf2173d36360dd6e19306d0afdc63c18e3af371a0e36df37a"),
    "damping": ("info/damp_probe.json", "907fe876506a25710ce12cc77a02abc5a22967964fa8859f8bf3e9e87f3559fa"),
    "budget_switch": ("budget_switch_probe.json",
                      "fcc7d4e6b583de763720651bd6e887d47de258bc6311f96dadbf7451def9c2a8"),
}
SCRATCH_SCRIPTS = {  # basename: sha256 of the exploratory script ported (session scratch, 2026-10-06)
    "posthoc_probe.py": "c263b894d6af29639ac11899ef5ba00796756591f9dc812652c540a9547fa409",
    "upd_probe.py": "3482c59642bdf7e0bf504053c780161a7db33498f7d8bb0c49a4173e10dfde38",
    "upd_probe2.py": "cc52e3ef5d4e2d60f7838c6e12a07e733283eb12612cbfa8e2b943fa11fa95d3",
    "b0_probe.py": "1be05ed7aa458caeebdaa58b77530e76a0b4de171190ce95d371e646b989efca",
    "info/damp_probe.py": "2de8341a68cc0bc6a75f0f08549ba5f55e0da44863c38be82b9ea90e540b6e4a",
    "budget_switch_probe.py": "113d73a1f7efbb6e9701538208532c837afc16baa953a2cfbe3d8cad7a7335ad",
    "common.py": "701bfb95f0e10f5d66422a12287f57ff43516ac0dfe9fbecf72aadfb4a7b01de",
    "posthoc_probe.out": "ddaf9e1fddab90b43055c027d019dd8f15122831849db15810ac01fb989eb321",
    "upd_probe.out": "83779fe1bc7cb75e35c7a20d6ff4b9b77e265e451fa3d49634d5c683f6153680",
    "upd_probe2.out": "a61259a4e7bace090db27d594cb1caba56f82f9e57f43c0f3d1458814c10b435",
    "b0_probe.out": "8498a7c500e61b42d5d7d10523194ec2d9ce27e60ebee12a907edaed1d7f467e",
    "ledger.md": "eae9169dc4bc97c282ea6baed2ee4d4a1a7938068c33ad5c4476b8dea59bf80e",
    "critic.md": "e7bddfbf390d2a9f46616d149b78362c9e86dad8dde4bf21a2b46f18e955155f",
}

_MAPS_LINE = re.compile(r"(\w+) \| " + r" \| ".join(rf"{r} ([+-]\d\.\d+) \((\d\.\d+)\)" for r in REGS))
_U_LINE = re.compile(r"(\S+) " + " ".join(rf"{r}:([+-]\d\.\d+)\((\d\.\d+)\)/hv([+-]\d\.\d+)" for r in REGS))


def printed_items():
    """(name, path, decimals, want) for every printed scratch number."""
    out = []
    for line in PRINTED["maps"].splitlines():
        m = _MAPS_LINE.fullmatch(line)
        kind = m.group(1)
        for j, reg in enumerate(REGS):
            out.append((("maps", "summary", kind, reg, "delta"), 5, float(m.group(2 + 2 * j))))
            out.append((("maps", "summary", kind, reg, "cluster_se"), 5, float(m.group(3 + 2 * j))))
    for block in ("update_speed", "update_conditional", "b0_stretch"):
        for line in PRINTED[block].splitlines():
            m = _U_LINE.fullmatch(line)
            key = m.group(1)
            for j, reg in enumerate(REGS):
                g = 2 + 3 * j
                out.append(((block, "levers", key, "score_raw", reg, "delta"), 5, float(m.group(g))))
                out.append(((block, "levers", key, "score_raw", reg, "se"), 5, float(m.group(g + 1))))
                out.append(((block, "levers", key, "score_hv", reg, "delta"), 5, float(m.group(g + 2))))
    for cell, (d, se) in PRINTED_DAMPING.items():
        mode, c = cell.split("/")
        out.append((("damping", mode, c, "delta_per_pair"), 5, d))
        out.append((("damping", mode, c, "se"), 5, se))
    for reg, g in PRINTED_SWITCH["in_sample"].items():
        out.append((("budget_switch", reg, "in_sample_switch_gain"), 5, g))
    return out


def _get(res, path):
    x = res
    for part in path:
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x


def _round_eq(got, want, d):
    return abs(round(float(got), d) - want) < 1e-9


def check_printed(res):
    items = []
    for path, d, want in printed_items() + [(p, d, w) for p, d, w, _ in STATEMENTS]:
        name = "/".join(str(p) for p in path)
        try:
            got = float(_get(res, path))
        except (KeyError, TypeError, IndexError):
            items.append({"path": name, "status": "not run", "want": want})
            continue
        ok = _round_eq(got, want, d)
        # 5-dp printed zeros: '-0.00000' and '+0.00000' are both 0 at 5 dp
        st = "match" if ok else ("explained" if name in EXPLAINED else "differs")
        it = {"path": name, "status": st, "got": got, "want": want, "decimals": d, "diff": got - want}
        if name in EXPLAINED:
            it["explanation"] = EXPLAINED[name]
        items.append(it)
    if res.get("budget_switch"):
        sched = res["budget_switch"]["robust_switch_all"]["schedule"]
        items.append({"path": "budget_switch/robust_switch_all/schedule",
                      "status": "match" if sched == PRINTED_SWITCH["robust_schedule"] else "differs",
                      "got": sched, "want": PRINTED_SWITCH["robust_schedule"]})
    return items


def _decimals_for(block, path):
    """The precision the probe stored a leaf at: None = full precision."""
    if block in ("update_speed", "update_conditional", "b0_stretch"):
        if "score_raw" in path or "score_hv" in path or (block == "update_speed" and path[:1] == ("baseline",)):
            return 5
        if block == "b0_stretch" and "b_bw_by_parent" in path:
            return 2
    if block == "damping" and path[-2:-1] == ("by_budget",):
        return 5
    return None


def compare_tree(block, mine, theirs, path=(), out=None):
    """Every leaf of the probe's stored JSON against this script's value."""
    out = [] if out is None else out
    if isinstance(theirs, dict):
        for k, v in theirs.items():
            if k in ("note", "secs"):
                continue
            m = None
            if isinstance(mine, dict):
                m = mine.get(k, mine.get(str(k)))
                if m is None and k.lstrip("-").isdigit():
                    m = mine.get(int(k))
            compare_tree(block, m, v, path + (k,), out)
    elif isinstance(theirs, list):
        if not isinstance(mine, (list, tuple)) or len(mine) != len(theirs):
            out.append({"path": "/".join(map(str, path)), "status": "differs", "got": mine, "want": theirs})
            return out
        for j, (m, v) in enumerate(zip(mine, theirs)):
            compare_tree(block, m, v, path + (j,), out)
    elif isinstance(theirs, bool) or isinstance(theirs, str) or theirs is None:
        out.append({"path": "/".join(map(str, path)), "status": "match" if mine == theirs else "differs",
                    "got": mine, "want": theirs, "kind": "exact"})
    else:
        if mine is None:
            out.append({"path": "/".join(map(str, path)), "status": "differs", "got": None, "want": theirs})
            return out
        d = _decimals_for(block, path)
        got = float(mine)
        if d is None:
            diff = abs(got - float(theirs))
            st = "match" if diff <= 1e-9 else ("match_4dp" if diff < 5e-5 else "differs")
        else:
            diff = abs(round(got, d) - float(theirs))
            st = "match" if diff < 1e-9 else ("match_4dp" if abs(got - float(theirs)) < 5e-5 else "differs")
        out.append({"path": "/".join(map(str, path)), "status": st, "got": got, "want": theirs,
                    "decimals": d, "diff": got - float(theirs)})
    return out


def compare_scratch_files(res, scratch_dir):
    if not scratch_dir:
        return None
    out = {}
    for block, (relp, digest) in SCRATCH_FILES.items():
        p = os.path.join(scratch_dir, relp)
        if not os.path.exists(p) or res.get(block) is None:
            out[block] = {"status": "not run"}
            continue
        with open(p) as f:
            theirs = json.load(f)
        if block == "maps":
            theirs = {k: theirs[k] for k in ("summary", "calibration", "n_appearances")}
        items = compare_tree(block, res[block], theirs)
        counts = {s: sum(1 for i in items if i["status"] == s) for s in ("match", "match_4dp", "differs")}
        num = [abs(i["diff"]) for i in items if isinstance(i.get("diff"), float)]
        out[block] = {"file": relp, "sha256": sha256(p), "sha256_expected": digest, "leaves": len(items),
                      "counts": counts, "max_abs_diff": max(num) if num else None,
                      "not_matching": [i for i in items if i["status"] != "match"][:50]}
    return out


# --- provenance, main ------------------------------------------------------------------------------

def rel(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(path, pattern):
    files = sorted(glob.glob(os.path.join(path, pattern)))
    h = hashlib.sha256()
    for f in files:
        h.update(f"{os.path.relpath(f, path)} {sha256(f)}\n".encode())
    return {"files": len(files), "sha256": h.hexdigest()}


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, got=None, want=None, required=True, note=None):
        self.items.append({"check": name, "ok": bool(ok), "got": got, "want": want,
                           "required": required, **({"note": note} if note else {})})
        return bool(ok)

    def failed(self):
        return [c for c in self.items if c["required"] and not c["ok"]]


def provenance(inputs, secs, libs):
    me = os.path.abspath(__file__)
    return {"command": " ".join(["python", rel(sys.argv[0])] + sys.argv[1:]),
            "script": rel(me), "script_sha256": sha256(me), "head": git("rev-parse", "HEAD"),
            "script_status": git("status", "--porcelain", "--", rel(me)) or "clean",
            "when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "inputs_sha256": inputs, "harness_rows_lib_digest": libs,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "seeds": SEEDS, "wall_s": round(secs, 1)}


def show(res):
    fam = res["family"]
    print(f"{'lever':<10} {'TL':>18} {'mix':>18} {'R1-bf':>18} {'R1-pu':>18}  verdict")
    for lever, r in fam["rows"].items():
        cells = "".join(f" {r['delta'][g]:+.5f} ({r['se'][g]:.5f})" for g in REGS)
        print(f"{lever:<10}{cells}  {r['verdict']}")
    b = fam["best_tl_lopo"]
    print(f"best LOPO TL delta over {fam['lopo_variants']} variants: {b['delta']:+.5f} ({b['se']:.5f}) {b['variant']}")
    if fam["budget_switch"]:
        s = fam["budget_switch"]
        print(f"budget switch: robust schedule {s['robust_schedule']}, worst regime {s['robust_worst_regime_gain']:+.6f}; "
              f"in sample {s['in_sample_gain_range'][0]:+.5f} to {s['in_sample_gain_range'][1]:+.5f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rows", default=HARNESS_ROWS, help="harness rows (experiments/harness.py --stage collect)")
    ap.add_argument("--rs-rows", default=RS_ROWS, help="regime-sensitivity rows")
    ap.add_argument("--scratch", default=None, help="session scratch directory with the probes' JSON outputs")
    args = ap.parse_args(argv)
    t0 = time.time()
    checks = Checks()
    base, libs = load_apps(args.rows)
    checks.add("harness rows: 650 runs, 5,371 appearances, one library", len(base) == 5371 and len(libs) == 1,
               {"appearances": len(base), "libs": libs}, {"appearances": 5371, "libs": 1})
    vs = hidden_v()
    sh = np.bincount([vbin(v) for v in vs], minlength=3) / len(vs)
    h = np.histogram(np.clip(vs, 0, 0.25), DAMP_BINS)[0]
    checks.add("26 formative pairs; both probes' hidden-v shares agree", len(vs) == 26 and
               np.allclose(sh, h / h.sum()), [len(vs), sh.tolist()], 26)
    res = {}
    for name, fn in (("maps", lambda: block_maps(base)),
                     ("update_speed", lambda: block_update_speed(base, vs, sh)),
                     ("update_conditional", lambda: block_update_conditional(base, sh)),
                     ("b0_stretch", lambda: block_b0_stretch(base, sh)),
                     ("damping", lambda: block_damping(base, vs)),
                     ("budget_switch", lambda: block_budget_switch(args.rs_rows))):
        t = time.time()
        res[name] = fn()
        print(f"{name}: {time.time() - t:.1f} s", flush=True)
    if res["budget_switch"]:
        ok = all(res["budget_switch"][r]["configs_in_rows"] == sorted(RS_CONFIGS) for r in RS_REGIMES)
        checks.add("RS rows hold exactly the 11 configurations the switch uses", ok)
    res["family"] = family(res)
    printed = check_printed(res)
    n = {s: sum(1 for i in printed if i["status"] == s) for s in ("match", "explained", "differs", "not run")}
    checks.add("every printed scratch value and stated number reproduces at its precision or is explained",
               n["differs"] == 0, n, None)
    files = compare_scratch_files(res, args.scratch)
    if files:
        for block, v in files.items():
            if v.get("status") == "not run":
                continue
            checks.add(f"{block}: the probe's stored JSON is the one recorded", v["sha256"] == v["sha256_expected"])
            checks.add(f"{block}: every stored scratch leaf reproduces (to 4 decimals at worst)",
                       v["counts"]["differs"] == 0, v["counts"], None)
    out = {"about": {
        "what": "post-hoc maps of hier-ship's stored predictions fitted leave one parent out, update-speed "
                "and B0 calibrations, and budget-wise switching between configurations (R5): every cheap "
                "lever on top of hier-ship, measured 2026-10-06, ported",
        "plan": "docs/plans/p2_final_plan.md section 2 (last bullet) and R5; the v2 outline's T4 row 6 and "
                "App F.8",
        "sign": "delta = ALC(lever) - ALC(hier-ship); negative means the lever is better",
        "regimes": {"tl": "test-like, seed 2 (300 runs)", "mix": "test-like mix/whole, seed 3 (150)",
                    "r1b": "public, benchmark-first, seed 0 (100)", "r1p": "public, pair-uniform, seed 0 (100)"},
        "modes": {"lopo": "fit on TL+mix appearances of the other parents (the primary reading)",
                  "lopo4": "fit on all four regimes' appearances of the other parents",
                  "ins": "fit and score in sample per regime (a ceiling, not a lever)"},
        "weights": "maps and damping fit with each run weighing 1; update_speed, update_conditional and "
                   "b0_stretch with each regime weighing 1 (w_raw) or hidden-v (w_hv); scores raw (run mean "
                   "of pairs) and hidden-v",
        "se": "(parent, subject) cluster bootstrap with the fit held fixed (a lower bound); 2,000 "
              "resamples in maps, 1,000 elsewhere",
        "verdict_rule": f"negative if the LOPO TL delta exceeds both 2 SE and {NEG_MIN}, or a public regime "
                        f"loses more than {GUARD_PUBLIC}; passes only at TL delta <= {GATE_TL} with the public "
                        f"guard held; null otherwise",
        "feedback_use": "results/formative/run1-3.txt B0 sections: 26 values v = B0 - ECE0^2 pooled into "
                        "three bin shares for the hidden-v weights; no id is read into the results",
        "harness_rows_library": "3f75a549673aae6a (archive-3's code: the corrected multiple-choice floor and "
                                "the floored-fit fix; docs/findings.md, the acceptance harness)",
        "budget_switch_rows": "data/regime_sensitivity_rows, seed 11 (experiments/regime_sensitivity.py)"},
        **res}
    out["checks"] = checks.items
    out["scratch"] = {"ported_from": SCRATCH_SCRIPTS, "files": {k: v[0] for k, v in SCRATCH_FILES.items()},
                      "printed_counts": n, "printed": printed, "explained": EXPLAINED,
                      "stored_json": files}
    inputs = {rel(p): sha256(p) for p in TABLES.values()}
    inputs[rel(args.rows) + "/*/*.pkl"] = dir_digest(args.rows, "*/*.pkl")
    inputs[rel(args.rs_rows) + "/*/*.json"] = dir_digest(args.rs_rows, "*/*.json")
    out["provenance"] = provenance(inputs, time.time() - t0, libs)
    show(out)
    print(f"printed scratch values: {n}")
    if files:
        for b, v in files.items():
            print(f"  {b}: {v.get('counts', v.get('status'))} max |diff| {v.get('max_abs_diff')}")
    for c in checks.items:
        if not c["ok"]:
            print(f"FAILED{' (required)' if c['required'] else ''}: {c['check']}: got {c['got']} want {c['want']}")
    for it in printed:
        if it["status"] in ("differs", "explained"):
            print(f"  {it['status']}: {it['path']} got {it.get('got')} want {it['want']}")
    if checks.failed():
        print("not written: a required check failed")
        return 1
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=float)
        f.write("\n")
    os.replace(tmp, args.out)
    print(f"wrote {rel(args.out)} ({out['provenance']['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
