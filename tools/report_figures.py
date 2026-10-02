"""Draw the technical report's figures from the committed results files.

    python tools/report_figures.py                 # every figure into docs/report/fig/
    python tools/report_figures.py --only gate_curve level_surface
    python tools/report_figures.py --out /tmp/fig  # somewhere else (its manifest goes there too)
    python tools/report_figures.py --check         # rebuild in a temporary directory and compare

Every figure reads `results/*.json` only, through `Inputs.load`, which refuses
any other path: nothing comes from `data/`, from session scratch, or from a
model run. Each figure is one function, `fig_<name>(inp) -> Figure`, and
`FIGURES` lists them, the review's priorities first (learning curves, the level
surface, the gate curve, regime sensitivity). Each is written as SVG (text as
paths) and as PNG at 200 dpi. The files are deterministic for a given
matplotlib: no dates in them, a fixed SVG hash salt, the bundled DejaVu Sans,
the Agg backend, and a seeded jitter where points are spread.

`docs/report/fig/manifest.json` records, per figure, the results files it read
with their sha256, the sha256 of each file written and the function that drew
it; and once, this script's sha256 and the library versions. `--check` (and
`tests/test_report_figures.py`) fails when an input or the script changed
after the figures were drawn, when a figure no longer builds, or, with the same
matplotlib, when a fresh build differs from the committed files. Captions are
in `docs/report/figures.md` ("Drawn figures"); they name the keys each panel
reads.

Light CPU only: the largest input (`results/level_calibration.json`, 19 MB) is
parsed once and shared by the figures that read it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report" / "fig"
SCRIPT = "tools/report_figures.py"
DPI = 200
WIDTH = 5.5  # inches: the NeurIPS text width

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
PARENTS = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]

# One colour per model in every figure (Okabe-Ito, colour-blind safe); every
# series also has its own marker and line style, so the panels read in grey.
C = {
    "ship": "#0072B2",     # shipped hier (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5)
    "legacy": "#D55E00",   # legacy Predictor (the first submission)
    "smooth": "#009E73",   # Beta(2,2) smoothed mean
    "aggr": "#E69F00",     # neighbouring configuration (mu0 -3.0, attr_scale 0.25)
    "eb_fit": "#CC79A7",   # hier with its fitted level prior (hier's defaults)
    "smcal": "#56B4E9",    # smoothed mean calibrated to the tuned regime
    "other": "#555555",
    "light": "#999999",
}
NAMES = {
    "ship": "shipped hier",
    "legacy": "legacy Predictor",
    "smooth": "smoothed Beta(2,2)",
    "aggr": "neighbouring config (−3.0 / 0.25)",
    "eb_fit": "hier, fitted level prior",
    "eb_adapt": "EB level on −3.0 / 0.5",
    "eb_ship": "EB level on ship's prior",
    "wide35": "sigma_mu 3.5",
    "wide50": "sigma_mu 5.0",
    "smcal": "calibrated smoothed mean",
    "onepl": "hier, no subject prior",
}
STYLE = {  # (colour, marker, line style)
    "ship": (C["ship"], "o", "-"),
    "legacy": (C["legacy"], "s", "--"),
    "smooth": (C["smooth"], "D", "-."),
    "aggr": (C["aggr"], "v", ":"),
    "eb_fit": (C["eb_fit"], "P", "-"),
    "eb_adapt": ("#000000", "d", "-"),
    "eb_ship": ("#555555", "d", "-"),
    "wide35": ("#555555", ">", "-"),
    "wide50": ("#555555", "<", "-"),
    "smcal": (C["smcal"], "h", "-"),
    "onepl": ("#555555", "X", "-"),
}
FORMATIVE = {  # run -> (marker, line style, label)
    1: ("^", (0, (4, 2)), "run 1: legacy Predictor"),
    2: ("o", "-", "run 2: shipped hier"),
    3: ("s", (0, (1, 1.5)), "run 3: shipped hier, rebuilt"),
}
PARENT_MARKER = dict(zip(PARENTS, ["o", "s", "^", "D"]))

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "svg.fonttype": "path",
    "svg.hashsalt": "paiec-report-figures",
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "hatch.linewidth": 0.5,
    "axes.titlepad": 3.0,
}


# ---------------------------------------------------------------------------
# inputs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    """Reads results files, and only those, and records which figure read which."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.cache: dict[str, object] = {}
        self.sha: dict[str, str] = {}
        self.used: list[str] = []

    @staticmethod
    def allowed(rel: str) -> bool:
        p = Path(rel)
        return (len(p.parts) == 2 and p.parts[0] == "results" and p.suffix == ".json"
                and not p.is_absolute())

    def begin(self):
        self.used = []

    def load(self, rel: str):
        if not self.allowed(rel):
            raise ValueError(f"figures read results/*.json only, not {rel!r}")
        if rel not in self.cache:
            path = self.root / rel
            self.sha[rel] = sha256_file(path)
            with open(path) as f:
                self.cache[rel] = json.load(f)
        if rel not in self.used:
            self.used.append(rel)
        return self.cache[rel]

    def record(self) -> dict[str, str]:
        return {rel: self.sha[rel] for rel in sorted(self.used)}


# ---------------------------------------------------------------------------
# helpers


def bx():
    """Budget positions on the log2(1 + n) axis, where they are equally spaced."""
    return np.log2(1 + np.asarray(BUDGETS, float))


BUDGET_LABEL = "budget B (labels revealed; log$_2$(1+B) axis)"


def budget_axis(ax, label=True):
    ax.set_xticks(bx())
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlim(-0.3, bx()[-1] + 0.3)
    if label:
        ax.set_xlabel(BUDGET_LABEL)


def alc(budgets):
    return float(np.dot(WEIGHTS, budgets))


def milli(ax, axis="y"):
    """Tick labels in units of 1e-3 (the data stay in ALC)."""
    f = FuncFormatter(lambda v, _: f"{v * 1e3:g}" if abs(v) > 1e-12 else "0")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def title(ax, letter, text, **kw):
    ax.set_title(f"$\\bf{{{letter}}}$  {text}", loc="left", **kw)


def widest(blocks: dict) -> str:
    """The span with the most runs (ties: the one listed first)."""
    return max(blocks, key=lambda s: blocks[s]["runs"])


# ---------------------------------------------------------------------------
# learning curves by regime (review m8, P2.20 item 1)

LC_REGIMES = [
    ("tl", "tuned test-like (seed 2)"),
    ("tl mix/whole", "test-like, mix/whole (seed 3)"),
    ("tl no shift", "test-like, no date shift (seed 3)"),
    ("r1b", "public R1, benchmark-first"),
    ("r1p", "public R1, pair-uniform"),
]


def formative_runs(inp):
    ff = inp.load("results/formative_feedback.json")
    f3 = inp.load("results/formative_run3.json")
    runs = {
        1: np.asarray(ff["record"]["run1"]["budgets"]["brier_mean"], float),
        2: np.asarray(ff["record"]["run2"]["budgets"]["brier_mean"], float),
        3: np.asarray(f3["run3"]["budgets"]["brier_mean"], float),
    }
    pairs = {1: ff["record"]["run1"]["pairs"], 2: ff["record"]["run2"]["pairs"],
             3: f3["run3"]["pairs"]}
    return runs, pairs


def learning_curve_series(inp):
    """Per regime: the shipped hier's mean and single-run sd by budget over its
    widest span, and the legacy Predictor's and the smoothed mean's means over
    the widest span each was matched on."""
    sc = inp.load("results/ship_confirm.json")
    out = {}
    for key, name in LC_REGIMES:
        reg = sc["regimes"][key]
        span = widest(reg["shipped"])
        ship = reg["shipped"][span]
        sd = sc["single_run_sd"][key]
        row = {"title": name, "ship_span": span, "ship_runs": ship["runs"],
               "ship": np.asarray(ship["budgets"], float),
               "sd": np.asarray([sd[f"B{b}"] for b in BUDGETS], float),
               "ship_ALC": ship["ALC"], "sd_ALC": sd["ALC"]}
        for comp in ("legacy", "smoothed"):
            blocks = reg["vs"].get(comp) or {}
            if blocks:
                s = widest(blocks)
                row[comp] = np.asarray([x["other"] for x in blocks[s]["budgets"]], float)
                row[comp + "_span"] = s
                row[comp + "_runs"] = blocks[s]["runs"]
        out[key] = row
    return out


def fig_learning_curves(inp):
    series = learning_curve_series(inp)
    runs, _ = formative_runs(inp)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.85), sharey=True)
    axes = axes.ravel()
    x = bx()
    for i, (key, _) in enumerate(LC_REGIMES):
        ax = axes[i]
        s = series[key]
        ax.fill_between(x, s["ship"] - s["sd"], s["ship"] + s["sd"], color=C["ship"],
                        alpha=0.16, lw=0, zorder=1)
        col, mk, ls = STYLE["ship"]
        ax.plot(x, s["ship"], color=col, marker=mk, ls=ls, zorder=3, ms=3)
        for comp, st in (("legacy", "legacy"), ("smoothed", "smooth")):
            if comp in s:
                col, mk, ls = STYLE[st]
                ax.plot(x, s[comp], color=col, marker=mk, ls=ls, zorder=3, ms=2.8)
        for r, b in runs.items():
            mk, ls, _ = FORMATIVE[r]
            ax.plot(x, b, color="k", marker=mk, ls=ls, lw=0.8, ms=3, mfc="white", mew=0.7,
                    zorder=4)
        title(ax, "abcde"[i], s["title"], fontsize=6.6)
        ax.text(0.97, 0.97, f"shipped: {s['ship_runs']} runs\nALC {s['ship_ALC']:.4f} "
                f"(sd {s['sd_ALC']:.3f})", transform=ax.transAxes, ha="right", va="top",
                fontsize=5.4, color=C["ship"])
        budget_axis(ax, label=False)
    ax = axes[5]
    ax.axis("off")
    handles = [
        (Line2D([], [], color=C["ship"], marker="o", ms=3), "shipped hier, mean over runs"),
        (Patch(color=C["ship"], alpha=0.16, lw=0), "shipped hier, ±1 single-run sd"),
        (Line2D([], [], color=C["legacy"], marker="s", ls="--", ms=2.8), NAMES["legacy"]),
        (Line2D([], [], color=C["smooth"], marker="D", ls="-.", ms=2.8), NAMES["smooth"]),
    ]
    for r in (1, 2, 3):
        mk, ls, lab = FORMATIVE[r]
        handles.append((Line2D([], [], color="k", marker=mk, ls=ls, lw=0.8, ms=3,
                               mfc="white", mew=0.7), lab))
    leg = ax.legend([h for h, _ in handles], [lab for _, lab in handles], loc="center left",
                    bbox_to_anchor=(-0.06, 0.5), handlelength=2.6, labelspacing=0.75,
                    fontsize=5.7, title="replica runs; formative runs (platform)",
                    title_fontsize=5.6)
    leg._legend_box.align = "left"
    axes[0].set_ylim(0.10, 0.38)
    fig.supxlabel(BUDGET_LABEL, fontsize=6.8, y=0.01)
    fig.supylabel("Brier score (mean over a run's pairs)", fontsize=6.8, x=0.01)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.11, hspace=0.32,
                        wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# the level-calibration surface

GRID_AS = ["0.25", "0.5", "0.75", "1.0"]
GRID_SM = ["sm=0.9", "sm=1.3", "sm=1.8", "sm=2.5"]


def lc_name(mu0, sm, a):
    return f"hier G mu0={mu0:+.2f} sm={sm:.2f} as={a:.2f}"


def parse_name(name):
    parts = dict(p.split("=") for p in name.split()[2:])
    return float(parts["mu0"]), float(parts["sm"]), float(parts["as"])


#: results/level_audit.json's `mild` blocks: the shipped configuration on the level
#: calibration's public runs 0-99 (seed 0), which level_calibration.json did not score
AUDIT_R1 = {"r1b": "r1b 0-99: public benchmark-first 0-99 (audit: fresh runs)",
            "r1p": "r1p 0-99: public pair-uniform 0-99 (audit: fresh runs)"}


def audit_guard(inp, name):
    """The worse public weighting of `name` against the legacy Predictor on the level
    calibration's R1 runs 0-99, from level_audit.json (None if it did not score it)."""
    mild = inp.load("results/level_audit.json")["mild"]
    g = []
    for r in ("r1b", "r1p"):
        cell = mild.get(AUDIT_R1[r], {}).get("vs PRED", {}).get(name)
        if cell is None or cell.get("runs") != 100:
            return None
        g.append(cell["minus ref (run / cluster / stratified SE)"][0])
    return max(g)


def surface_data(inp):
    lc = inp.load("results/level_calibration.json")
    s = lc["summary"]
    ship_name = lc_name(-2.5, 2.5, 0.5)
    ship_guard = audit_guard(inp, ship_name)
    gs = s["grid_surface"]
    mu = [float(m) for m in gs["mu0"]]
    A = np.full((len(GRID_AS), len(mu)), np.nan)
    G = np.full_like(A, np.nan)
    src = np.full(A.shape, "", dtype=object)
    for i, a in enumerate(GRID_AS):
        row = gs[f"as={a}"]["sm=2.5"]
        for j, m in enumerate(mu):
            if row[j] is not None:
                A[i, j] = row[j]
            name = lc_name(m, 2.5, float(a))
            g = [s["r1"][r].get(name, {}).get("diff", {}).get("mean") for r in ("r1b", "r1p")]
            if all(v is not None for v in g):
                G[i, j], src[i, j] = max(g), "r1"
            elif name == ship_name and ship_guard is not None:
                G[i, j], src[i, j] = ship_guard, "audit"
            elif s["shortlist"]["screen"].get(name) is not None:
                G[i, j], src[i, j] = s["shortlist"]["screen"][name], "screen"
    gap = {}
    for sm in GRID_SM[:-1]:
        vals = []
        for a in GRID_AS:
            for j in range(len(mu)):
                v, ref = gs[f"as={a}"][sm][j], gs[f"as={a}"]["sm=2.5"][j]
                if v is not None and ref is not None:
                    vals.append(v - ref)
        gap[sm] = np.asarray(vals, float)
    sel = s["selection"]["configs"]
    return {"mu": mu, "alc": A, "guard": G, "guard_src": src, "sm_gap": gap,
            "marks": {"chosen": s["selected"]["chosen"], "aggr": s["ship"]["config"],
                      "ship": ship_name},
            "hier_default": sel["hier"]["ALC"][0], "legacy": sel["Predictor"]["ALC"][0],
            "runs": s["selection"]["runs"], "guard_bar": lc["config"]["guard"]}


def fig_level_surface(inp):
    d = surface_data(inp)
    mu, A, G = d["mu"], d["alc"], d["guard"]
    fig = plt.figure(figsize=(WIDTH, 2.85))
    ax = fig.add_axes([0.07, 0.337, 0.6, 0.587])
    cax = fig.add_axes([0.68, 0.337, 0.012, 0.587])
    ax2 = fig.add_axes([0.865, 0.337, 0.125, 0.587])
    cmap = plt.get_cmap("cividis_r").copy()
    cmap.set_bad("#f2f2f2")
    vmin, vmax = np.nanmin(A), np.nanmax(A)
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   origin="lower", interpolation="nearest")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isnan(A[i, j]):
                ax.text(j, i, "not\nscored", ha="center", va="center", fontsize=4.4,
                        color="#9a9a9a")
                continue
            dark = (A[i, j] - vmin) / (vmax - vmin) > 0.55
            tc = "white" if dark else "black"
            ax.text(j, i + 0.17, f"{A[i, j]:.4f}", ha="center", va="center", fontsize=4.9,
                    color=tc)
            if not np.isnan(G[i, j]):
                tag = {"r1": "", "audit": "ᵃ"}.get(d["guard_src"][i, j], "ˢ")
                ax.text(j, i - 0.2, f"{G[i, j]:+.4f}{tag}", ha="center", va="center",
                        fontsize=4.3, color=tc)
                if G[i, j] > d["guard_bar"]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor="#c8c8c8" if dark else "#5a5a5a", lw=0))
    marks = [("chosen", "^", "k", "the rule's choice"),
             ("aggr", "D", C["aggr"], "neighbouring config (recommended)"),
             ("ship", "*", C["ship"], "shipped config (after the audit)")]
    handles, labels = [], []
    for key, mk, col, lab in marks:
        m, _, a = parse_name(d["marks"][key])
        j, i = mu.index(m), [float(v) for v in GRID_AS].index(a)
        ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, lw=1.3,
                               edgecolor=col))
        ax.plot(j - 0.34, i + 0.32, marker=mk, ms=6 if mk == "*" else 3.8, color=col,
                mec="white", mew=0.4, zorder=5, ls="")
        handles.append(Line2D([], [], ls="", marker=mk, ms=6 if mk == "*" else 3.8, color=col))
        labels.append(f"{lab} ({m:.1f} / {a:g})".replace("-", "−"))
    handles.append(Patch(facecolor="white", edgecolor="#5a5a5a", hatch="////", lw=0.4))
    labels.append(f"public guard fails (cost > +{d['guard_bar']:.3f})")
    handles.append(Line2D([], [], ls="", marker="$x$", color="none"))
    labels.append("lower number: public cost vs legacy Predictor, worse of two weightings "
                  "(100 runs each;\nᵃ: the same 100 runs, scored by the level audit; "
                  "ˢ: 40-run benchmark-first screen)")
    ax.set_xticks(range(len(mu)))
    ax.set_xticklabels([f"{m:g}" for m in mu])
    ax.set_yticks(range(len(GRID_AS)))
    ax.set_yticklabels(GRID_AS)
    ax.set_xlabel("mu0: centre of a new benchmark's level prior (logit)")
    ax.set_ylabel("attr_scale")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    leg = ax.legend(handles[:4], labels[:4], loc="upper left", bbox_to_anchor=(-0.02, -0.19),
                    ncol=2, fontsize=5.4, columnspacing=1.0, handletextpad=0.3,
                    labelspacing=0.35)
    ax.add_artist(leg)
    ax.text(0.0, -0.43, labels[4], transform=ax.transAxes, fontsize=5.2, va="top")
    title(ax, "a", f"test-like selection-half ALC, sigma_mu 2.5 ({d['runs']} runs)",
          fontsize=6.4)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5.2, length=2)
    cb.set_label("ALC", fontsize=5.8, labelpad=1)
    cb.outline.set_linewidth(0.4)

    rng = np.random.default_rng(0)
    for k, sm in enumerate(GRID_SM[:-1]):
        v = d["sm_gap"][sm]
        jit = (rng.random(len(v)) - 0.5) * 0.4
        ax2.plot(np.full(len(v), k) + jit, v, ls="", marker="o", ms=2.0, color=C["other"],
                 alpha=0.75, mew=0)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([s.split("=")[1] for s in GRID_SM[:-1]])
    ax2.set_xlabel("sigma_mu")
    ax2.set_ylabel("ALC(sigma_mu) − ALC(2.5), same cell (10$^{-3}$)", fontsize=5.8)
    ax2.set_xlim(-0.6, 2.6)
    milli(ax2)
    title(ax2, "b", "other widths", fontsize=6.4)
    return fig


# ---------------------------------------------------------------------------
# the acceptance gate, with the measured covariates on it

GATE_LINES = [("transferred nested", "transferred slope", C["ship"], "o", "-"),
              ("per-pair nested", "per-pair slope", C["aggr"], "s", "--")]


def gate_table(inp):
    h = inp.load("results/harness_thresholds.json")
    keys = sorted(h["tables"]["honest"], key=lambda k: float(k.split("=")[1]))
    out = {"r": [float(k.split("=")[1]) for k in keys],
           "rw": [h["tables"]["honest"][k]["meta"]["r_within_pair_tl"] for k in keys],
           "lines": {}, "gate": h["meta"]["gate"]}
    for line, *_ in GATE_LINES:
        th = h["thresholds"]["honest"][line]
        rows = [th[k] for k in keys]
        out["lines"][line] = {
            "tl": np.asarray([r["tl"] for r in rows], float),
            "se": np.asarray([r["tl_cluster_se"] for r in rows], float),
            "draw_sd": np.asarray([r["tl_draw_sd"] for r in rows], float),
            "pass_draws": [r["pass_draws"] for r in rows],
            "reps": [h["tables"]["honest"][k]["lines"][line].get("replicates", {}).get("tl", [])
                     for k in keys],
            "majority": th["smallest_r_majority_of_draws"],
            "every": th["smallest_r_every_draw"],
        }
    o = h["acceptance"]["oracle_honest"]["transferred nested"]["regimes"]["tl"]
    out["oracle"] = (o["est"], o["cluster_se"])
    return out


FAMILIES = {  # family -> (colour, marker)
    "4B judge (Qwen3-4B)": ("#D55E00", "v"),
    "4B entropy and hidden-state probes": ("#CC79A7", "^"),
    "14B rubric features and heads (Qwen3-14B)": ("#009E73", "s"),
    "14B reasoning entropy (commit D)": ("#0072B2", "D"),
    "embedding heads (frozen, fine-tuned)": ("#E69F00", "o"),
    "known-sign item cues": ("#555555", "P"),
    "TF-IDF": ("#000000", "x"),
}
COVARIATES = [  # family, file, path to the harness block, covariates
    ("4B judge (Qwen3-4B)", "results/llm4b_close.json", ("harness",),
     ["rating", "digit_mode", "entropy", "nll"]),
    ("4B entropy and hidden-state probes", "results/hidden_state_probe.json", ("harness",),
     ["entropy", "surprisal", "profile", "hidden", "hidden_mean", "hidden_L9", "hidden_L18",
      "hidden_L27", "hidden_L36"]),
    ("14B rubric features and heads (Qwen3-14B)", "results/strong_llm_eval.json", ("harness",),
     ["head rubric_ridge", "head rubric_ridge_posfree", "head rubric_all_ridge",
      "head judge_ridge", "rubric_sum", "rubric_reasoning", "rubric_knowledge", "rubric_work",
      "rubric_interaction", "rubric_volume", "rubric_atypicality", "rubric_precision",
      "rubric_unguessability", "solve_share", "time_log_minutes"]),
    ("14B reasoning entropy (commit D)", "results/strong_llm_eval.json", ("entropy", "harness"),
     ["ent_first1024", "ent_first256", "lp_first1024", "lp_first256", "ent_n_tokens",
      "ent_closed"]),
    ("embedding heads (frozen, fine-tuned)", "results/finetune_encoder.json", ("harness",),
     ["frozen_nested", "frozen_lobo512", "finetuned", "finetuned_e3"]),
    ("known-sign item cues", "results/itemcov_eval.json", ("harness",),
     ["stated_size", "position", "position_within", "format_score", "log_length", "image_ref"]),
]


def covariate_points(inp):
    """Each covariate's within-pair r and its nested test-like line (transferred
    slope; for the known-sign cues, the nested selection over their allowed
    forms), with the cluster SE."""
    pts = []
    for fam, f, path, names in COVARIATES:
        h = inp.load(f)
        for p in path:
            h = h[p]
        for n in names:
            c = h[n]
            lines = c["lines"]
            if "transferred nested" in lines:
                ln = lines["transferred nested"]
                est, se = ln["tl"], ln.get("tl_cluster_se", 0.0)
                pp = lines.get("per-pair nested", {}).get("tl")
            else:
                ln = lines["allowed nested"]["regimes"]["tl"]
                est, se = ln["est"], ln["cluster_se"]
                pp = None
            pts.append({"family": fam, "name": n, "r": abs(c["r_within_pair_tl"]),
                        "r_signed": c["r_within_pair_tl"], "est": est, "se": se,
                        "per_pair": pp, "coverage": c.get("coverage_eval_items")})
    return pts


TRANSFER = [  # row label, family, file, kind
    ("TF-IDF+SVD ridge, leave one benchmark out", "TF-IDF", "results/emb_transfer.json",
     "tfidf_ridge_centred"),
    ("embedding ridge, leave one benchmark out", "embedding heads (frozen, fine-tuned)",
     "results/emb_transfer.json", "emb_ridge_centred"),
    ("embedding kNN, leave one benchmark out", "embedding heads (frozen, fine-tuned)",
     "results/emb_transfer.json", "emb_knn_centred"),
    ("4B judge rating (sign as declared)", "4B judge (Qwen3-4B)", "results/llm4b_close.json",
     "judge4b"),
    ("14B rubric head (primary), out of fold", "14B rubric features and heads (Qwen3-14B)",
     "results/strong_llm_eval.json", "head14b"),
    ("14B ent_first1024 (primary)", "14B reasoning entropy (commit D)",
     "results/strong_llm_eval.json", "ent14b"),
]


def transfer_points(inp):
    """Per benchmark, Pearson r of a text-read signal with item difficulty, with
    its 95% group-bootstrap CI, oriented by the declared sign where one is."""
    out = []
    for lab, fam, f, kind in TRANSFER:
        d = inp.load(f)
        for p in PARENTS:
            if kind.endswith("_centred"):
                u = d["lobo"][kind][p]
                est, ci = u["pearson"], list(u["pearson_ci_groupboot"])
            elif kind == "judge4b":
                feat = d["signs"]["features"]["rating"]
                u = feat["units"].get(p)
                if u is None or "pearson" not in u or d["signs"]["coverage"][p]["rated"] == 0:
                    continue
                sg = feat["declared_sign"]
                est, ci = sg * u["pearson"]["est"], sorted(sg * v for v in u["pearson"]["ci_group"])
            elif kind == "head14b":
                u = d["heads"]["rubric_ridge"]["per_parent"][p]["pearson"]
                est, ci = u["est"], list(u["ci_group"])
            else:
                feat = d["entropy"]["signs"]["features"]["ent_first1024"]
                u = feat["units"][p]["pearson"]
                sg = feat["declared_sign"]
                est, ci = sg * u["est"], sorted(sg * v for v in u["ci_group"])
            out.append({"label": lab, "family": fam, "parent": p, "est": est, "ci": ci})
    return out


def fig_gate_curve(inp):
    t = gate_table(inp)
    pts = covariate_points(inp)
    tp = transfer_points(inp)
    gate = t["gate"]["tl"]
    fig = plt.figure(figsize=(WIDTH, 5.5))
    ax = fig.add_axes([0.095, 0.585, 0.37, 0.345])
    bx_ = fig.add_axes([0.6, 0.585, 0.385, 0.345])
    cx = fig.add_axes([0.385, 0.075, 0.6, 0.36])

    # (a) the gate curve against the honest r of a synthetic covariate
    r = np.asarray(t["r"])
    for line, lab, col, mk, ls in GATE_LINES:
        L = t["lines"][line]
        ax.fill_between(r, L["tl"] - L["se"], L["tl"] + L["se"], color=col, alpha=0.2, lw=0)
        ax.plot(r, L["tl"], color=col, marker=mk, ls=ls, ms=3, label=f"{lab}, nested", zorder=3)
        dx = 0.014 if line.startswith("per") else -0.014
        for k, reps in enumerate(L["reps"]):
            if reps:
                ax.plot(np.full(len(reps), r[k] + dx), reps, ls="", marker=".", ms=2.0,
                        color=col, alpha=0.55, zorder=2)
    ax.axhline(gate, color="k", lw=0.8, ls=(0, (5, 2)))
    ax.axhline(0, color="#888888", lw=0.5)
    ax.text(0.0, gate - 0.0004, f"gate {gate * 1e3:g}", fontsize=5.6, ha="left", va="top")
    L = t["lines"]["transferred nested"]
    for k, pd in enumerate(L["pass_draws"]):
        if r[k] >= 0.2:
            ax.annotate(pd, (r[k], L["tl"][k]), xytext=(-4, -8), textcoords="offset points",
                        fontsize=5.0, color=C["ship"], ha="right")
    ox, ose = t["oracle"]
    ax.annotate(f"honest oracle {ox * 1e3:.1f} ± {ose * 1e3:.1f}\n(off scale)",
                xy=(0.745, -0.0168), xytext=(0.30, -0.0158), fontsize=5.2, va="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color="k"))
    ax.set_xlim(-0.03, 0.76)
    ax.set_ylim(-0.0172, 0.0012)
    milli(ax)
    ax.set_xlabel("honest r of a synthetic covariate")
    ax.set_ylabel("test-like ALC difference vs shipped hier (10$^{-3}$)")
    sec = ax.secondary_xaxis("top", functions=(
        lambda v: np.interp(v, t["r"], t["rw"]), lambda v: np.interp(v, t["rw"], t["r"])))
    sec.set_xticks([round(v, 2) for v in t["rw"]][1:])
    sec.tick_params(labelsize=5.0, pad=1)
    sec.set_xlabel("its within-pair r (test-like)", fontsize=5.8, labelpad=2)
    ax.legend(loc="lower left", fontsize=5.6, bbox_to_anchor=(0.0, 0.2))
    ax.text(-0.22, 1.13, "$\\bf{a}$  synthetic covariates", transform=ax.transAxes,
            fontsize=6.4)

    # (b) zoom near the gate: the measured covariates at their |within-pair r|
    ax = bx_
    rw = np.asarray(t["rw"])
    for line, lab, col, mk, ls in GATE_LINES:
        L = t["lines"][line]
        ax.fill_between(rw, L["tl"] - L["se"], L["tl"] + L["se"], color=col, alpha=0.15, lw=0)
        ax.plot(rw, L["tl"], color=col, ls=ls, lw=0.9, zorder=2)
    ax.axhline(gate, color="k", lw=0.8, ls=(0, (5, 2)))
    ax.axhline(0, color="#888888", lw=0.5)
    handles = []
    for fam, (col, mk) in FAMILIES.items():
        P = [p for p in pts if p["family"] == fam]
        if not P:
            continue
        ax.errorbar([p["r"] for p in P], [p["est"] for p in P], yerr=[p["se"] for p in P],
                    ls="", marker=mk, ms=3.2, color=col, mec="white", mew=0.3,
                    elinewidth=0.6, capsize=0, zorder=4, alpha=0.75)
        handles.append((Line2D([], [], ls="", marker=mk, ms=3.2, color=col), fam))
    ss = next(p for p in pts if p["name"] == "stated_size")
    ax.annotate("stated_size\n(on one parent)", (ss["r"], ss["est"]), xytext=(-3, 8),
                textcoords="offset points", fontsize=4.9, ha="center")
    ax.set_xlim(-0.01, 0.45)
    ax.set_ylim(-0.0042, 0.0019)
    milli(ax)
    ax.set_xlabel("|within-pair r| of a measured covariate (test-like)")
    ax.set_ylabel("nested test-like ALC difference (10$^{-3}$)")
    ax.legend([h for h, _ in handles], [f for _, f in handles], loc="lower left", fontsize=4.9,
              handletextpad=0.1, labelspacing=0.25, borderaxespad=0.1)
    title(ax, "b", "measured covariates, ±1 cluster SE", fontsize=6.3)

    # (c) per benchmark: correlation of text-read signals with item difficulty
    labels = [lab for lab, *_ in TRANSFER]
    ypos = {lab: k for k, lab in enumerate(labels)}
    off = {p: (q - 1.5) * 0.17 for q, p in enumerate(PARENTS)}
    lo, hi = t["lines"]["transferred nested"]["majority"], t["lines"]["transferred nested"]["every"]
    cx.axvspan(lo, hi, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
    cx.text((lo + hi) / 2, -0.95, f"transferred slope passes\nmost draws from r {lo:g},\n"
            f"every draw from {hi:g}", fontsize=4.6, ha="center", va="center")
    cx.axvline(0, color="#888888", lw=0.5)
    for q in tp:
        col = FAMILIES[q["family"]][0]
        y = ypos[q["label"]] + off[q["parent"]]
        cx.plot(q["ci"], [y, y], color=col, lw=0.7, alpha=0.9)
        cx.plot(q["est"], y, marker=PARENT_MARKER[q["parent"]], ls="", color=col, ms=3.0,
                mec="white", mew=0.3)
    cx.set_yticks(range(len(labels)))
    cx.set_yticklabels(labels, fontsize=5.6)
    cx.set_ylim(len(labels) - 0.5, -1.45)
    cx.set_xlim(-0.55, 0.62)
    cx.grid(axis="y", visible=False)
    cx.set_xlabel("Pearson r with item difficulty, per benchmark (95% group-bootstrap CI)")
    cx.legend([Line2D([], [], ls="", marker=PARENT_MARKER[p], color="#444444", ms=3)
               for p in PARENTS], PARENTS, loc="lower left", fontsize=5.0, ncol=1,
              handletextpad=0.1, labelspacing=0.25, borderaxespad=0.2)
    cx.text(-0.62, 1.04, "$\\bf{c}$", transform=cx.transAxes, fontsize=7.5)
    cx.text(-0.62, 1.04, "     signals read off the item text, one marker per benchmark",
            transform=cx.transAxes, fontsize=6.3)
    return fig


# ---------------------------------------------------------------------------
# regime sensitivity (P1a)

RS_CONFIGS = ["aggr", "eb_adapt", "eb_ship", "wide35", "wide50", "eb_fit", "onepl", "smcal",
              "smooth", "legacy"]
RS_XLIM = (-0.0065, 0.0125)
RS_NAMES = {"aggr": "neighbour (−3.0 / 0.25)", "onepl": "hier, no subject prior",
            "smcal": "smoothed, calibrated"}


def regime_sensitivity_data(inp):
    d = inp.load("results/regime_sensitivity.json")
    s = d["summary"]
    tl = sorted((r for r in s["vs_ship"] if not r.startswith("R1")),
                key=lambda r: s["realised"][r]["mean"])
    order = tl + [r for r in ("R1B", "R1P") if r in s["vs_ship"]]
    rows = {r: {c: {"D": s["vs_ship"][r][c]["exact"]["D"],
                    "se": s["vs_ship"][r][c]["exact"]["cluster_se"],
                    "U95": s["vs_ship"][r][c]["exact"]["U95"],
                    "PL": s["vs_ship"][r][c]["table"]["parent_level"]["mean"]}
                for c in RS_CONFIGS} for r in order}
    return {"order": order, "rows": rows, "realised": s["realised"],
            "th": d["rule"]["thresholds"], "outcome": d["rule"]["outcome"],
            "ship_alc": {r: s["configs"][r]["ship"]["ALC"] for r in order},
            "runs": {r: s["configs"][r]["ship"]["runs"] for r in order}}


def fig_regime_sensitivity(inp):
    d = regime_sensitivity_data(inp)
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 4.5), sharey=True)
    axes = axes.ravel()
    ys = np.arange(len(RS_CONFIGS))
    lo, hi = RS_XLIM
    th = d["th"]
    for k, reg in enumerate(d["order"]):
        ax = axes[k]
        for y, c in zip(ys, RS_CONFIGS):
            v = d["rows"][reg][c]
            col, mk, _ = STYLE[c]
            D, se = v["D"], v["se"]
            if lo < D < hi:
                ax.plot([max(D - 1.96 * se, lo), min(D + 1.96 * se, hi)], [y, y], color=col,
                        lw=0.6, solid_capstyle="butt")
                ax.plot([max(D - se, lo), min(D + se, hi)], [y, y], color=col, lw=2.0,
                        solid_capstyle="butt", alpha=0.8)
                ax.plot(D, y, marker=mk, color=col, ms=3.0, mec="white", mew=0.3, zorder=4)
            else:
                edge = hi if D >= hi else lo
                sgn = 1 if D >= hi else -1
                ax.annotate("", xy=(edge, y), xytext=(edge - 0.0022 * sgn, y),
                            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=col))
                ax.text(edge - 0.0025 * sgn, y, f"{D * 1e3:+.0f}", fontsize=4.9,
                        ha="right" if sgn > 0 else "left", va="center", color=col)
            if lo < v["PL"] < hi:
                ax.plot(v["PL"], y + 0.32, marker="o", ms=2.3, mfc="white", mec=col, mew=0.6,
                        ls="", zorder=3)
        ax.axvline(0, color="#888888", lw=0.6)
        if reg in ("READING", "AUDIT"):
            ax.axvline(-th["gain"], color="k", lw=0.7, ls=(0, (4, 2)))
        elif reg.startswith("R1"):
            ax.axvline(th["public_loss"], color="k", lw=0.7, ls=(0, (1, 1.5)))
        else:
            ax.axvline(th["testlike_loss"], color="k", lw=0.7, ls=(0, (1, 1.5)))
        re_ = d["realised"][reg]
        ax.set_title(f"$\\bf{{{'abcdefg'[k]}}}$ {reg}\nlevel {re_['mean']:+.2f}, sd {re_['sd']:.2f}",
                     fontsize=6.0, loc="left")
        ax.set_xlim(lo, hi)
        ax.set_xticks([-0.004, 0, 0.004, 0.008, 0.012])
        milli(ax, "x")
        ax.tick_params(axis="x", labelsize=5.6)
        ax.grid(axis="y", visible=False)
        if k % 4 == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels([RS_NAMES.get(c, NAMES[c]) for c in RS_CONFIGS], fontsize=5.6)
    axes[0].set_ylim(len(RS_CONFIGS) - 0.4, -0.6)
    ax = axes[-1]
    ax.axis("off")
    handles = [
        (Line2D([], [], color="#555555", lw=2.0), "±1 cluster SE"),
        (Line2D([], [], color="#555555", lw=0.6), "±1.96 cluster SE"),
        (Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec="#555555"),
         "parent-level mean"),
        (Line2D([], [], color="k", lw=0.7, ls=(0, (4, 2))), f"rule's gain bar −{th['gain']:g}"),
        (Line2D([], [], color="k", lw=0.7, ls=(0, (1, 1.5))),
         f"loss bound +{th['testlike_loss']:g}\n(public +{th['public_loss']:g})"),
        (Line2D([], [], color="#555555", lw=0.8, marker=">", ms=3), "off scale (value, 10$^{-3}$)"),
    ]
    ax.legend([h for h, _ in handles], [lab for _, lab in handles], loc="center left",
              bbox_to_anchor=(-0.12, 0.5), fontsize=5.5, labelspacing=0.8)
    fig.supxlabel("ALC: config − shipped hier (10$^{-3}$; negative: config better)",
                  fontsize=6.6, y=0.01)
    fig.subplots_adjust(left=0.195, right=0.985, top=0.93, bottom=0.09, hspace=0.35,
                        wspace=0.12)
    return fig


# ---------------------------------------------------------------------------
# the formative runs, pair by pair


def fig_formative_runs(inp):
    runs, pairs = formative_runs(inp)
    series = learning_curve_series(inp)
    f3 = inp.load("results/formative_run3.json")
    models = {t_["run"]: t_ for t_ in f3["three_runs"]["table"]}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.45), sharey=True)
    x = bx()
    for k, r in enumerate((1, 2, 3)):
        ax = axes[k]
        model = "legacy" if r == 1 else "ship"
        for key, ls, a in (("tl", "-", 0.14), ("r1b", "--", 0.07)):
            s = series[key]
            mean = s["legacy"] if model == "legacy" else s["ship"]
            if model == "ship":
                ax.fill_between(x, mean - s["sd"], mean + s["sd"], color=C["ship"], alpha=a,
                                lw=0)
            ax.plot(x, mean, color=C[model], ls=ls, lw=0.9)
        for p in pairs[r]:
            ax.plot(x, p["brier"], color="#808080", lw=0.5)
        mk, _, _ = FORMATIVE[r]
        ax.plot(x, runs[r], color="k", lw=1.5, marker=mk, ms=3.2, mfc="white", mew=0.8)
        m = models[r]
        who = "legacy Predictor" if r == 1 else ("shipped hier" if r == 2 else
                                                 "shipped hier, rebuilt")
        title(ax, "abc"[k], f"run {r}: {who}\n{m['pairs']} pairs, {m['benchmarks']} benchmarks, "
              f"ALC {alc(runs[r]):.4f}", fontsize=6.2)
        budget_axis(ax, label=False)
    axes[0].set_ylabel("Brier score")
    axes[0].set_ylim(0, 0.6)
    handles = [
        (Line2D([], [], color="#808080", lw=0.5), "one pair"),
        (Line2D([], [], color="k", lw=1.5), "the run's mean"),
        (Line2D([], [], color=C["legacy"], lw=0.9),
         "legacy Predictor: tuned test-like (solid), public R1 (dashed)"),
        (Line2D([], [], color=C["ship"], lw=0.9),
         "shipped hier: tuned test-like ±1 sd (solid), public R1 ±1 sd (dashed)"),
    ]
    fig.legend([h for h, _ in handles], [lab for _, lab in handles], loc="lower center",
               ncol=2, fontsize=5.5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.supxlabel(BUDGET_LABEL, fontsize=6.6, y=0.12)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.3, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# what the level fix trades, budget by budget

LF_REGIMES = [("tl", "tuned test-like", "#0072B2", "o", "-"),
              ("tl mix/whole", "mix/whole", "#56B4E9", "s", "-"),
              ("tl no shift", "no date shift", "#000000", "D", ":"),
              ("r1b", "public, benchmark-first", "#777777", "^", "--"),
              ("r1p", "public, pair-uniform", "#AAAAAA", "v", "--")]
LF_CONFIGS = [("ship", lc_name(-2.5, 2.5, 0.5)), ("aggr", lc_name(-3.0, 2.5, 0.25)),
              ("eb_fit", "hier")]


def fig_level_fix_budgets(inp):
    sc = inp.load("results/ship_confirm.json")
    lc = inp.load("results/level_calibration.json")["summary"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.45), sharey=True)
    x = bx()
    n = len(LF_REGIMES)
    for k, (key, lab, col, mk, ls) in enumerate(LF_REGIMES):
        blocks = sc["regimes"][key]["vs"]["legacy"]
        span = widest(blocks)
        b = blocks[span]["budgets"]
        dx = (k - (n - 1) / 2) * 0.07
        ax.errorbar(x + dx, [v["diff"] for v in b], yerr=[v["cluster_se"] for v in b],
                    color=col, marker=mk, ms=2.8, lw=0.9, ls=ls, elinewidth=0.7, capsize=0,
                    label=f"{lab} ({blocks[span]['runs']} runs)")
    ax.axhline(0, color="#888888", lw=0.6)
    budget_axis(ax)
    ax.set_ylabel("Brier: config − legacy Predictor")
    ax.legend(loc="lower right", fontsize=5.4)
    title(ax, "a", "shipped config, ±1 cluster SE", fontsize=6.4)

    for key, name in LF_CONFIGS:
        col, mk, _ = STYLE[key]
        sel = lc["selection"]["configs"][name]["diff_budgets"]
        ax2.plot(x, sel, color=col, marker=mk, ms=2.8, ls="-", lw=1.0)
        if key == "ship":  # not in the level calibration's guard: the same 100 runs
            pub = [v["diff"] for v in sc["regimes"]["r1b"]["vs"]["legacy"]["0-99"]["budgets"]]
        else:
            pub = lc["r1"]["r1b"][name]["diff_budgets"]
        ax2.plot(x, pub, color=col, marker=mk, ms=2.8, ls="--", lw=1.0, mfc="white")
    ax2.axhline(0, color="#888888", lw=0.6)
    budget_axis(ax2)
    handles = [(Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1], ms=2.8), NAMES[k])
               for k, _ in LF_CONFIGS]
    handles += [(Line2D([], [], color="#555555", ls="-"), "test-like, selection half (100 runs)"),
                (Line2D([], [], color="#555555", ls="--", marker="o", mfc="white", ms=2.8),
                 "public, benchmark-first (100 runs)")]
    ax2.legend([h for h, _ in handles], [lab for _, lab in handles], loc="lower right",
               fontsize=5.3)
    title(ax2, "b", "three configs, level-calibration runs", fontsize=6.4)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.91, bottom=0.17, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# the empirical mean's ALC follows from base rates


def analytic_line(formula: str):
    m = re.match(r"\s*([0-9.]+)\s*\+\s*([0-9.]+)\s*\*", formula)
    if not m:
        raise ValueError(f"unexpected analytic formula {formula!r}")
    return float(m.group(1)), float(m.group(2))


def fig_empirical_mean(inp):
    r1 = inp.load("results/official_baselines.json")["r1"]
    q = np.asarray([row["q"] for row in r1["per_run"]], float)
    y = np.asarray([row["alc"]["empirical mean"] for row in r1["per_run"]], float)
    an = r1["analytic"]
    a, b = analytic_line(an["formula"])
    entry = r1["leaderboard"]["entries"]["organisers' entry"]
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, 2.45))
    ax.plot(q, y, ls="", marker="o", ms=1.8, alpha=0.45, color=C["other"], mew=0,
            label=f"one R1 run ({len(q)} runs)")
    xx = np.linspace(0.07, q.max() * 1.03, 50)
    ax.plot(xx, a + b * xx, color=C["ship"], lw=1.0, label=f"{a:g} + {b:g} E[p(1−p)]")
    ax.errorbar(an["E_q"][0], an["observed"][0], yerr=1.96 * an["observed"][1], marker="s",
                ms=3.5, color="k", ls="", capsize=0, label="mean over runs (±1.96 SE)")
    ax.plot(an["E_q"][0], an["exact"][0], marker="D", ms=3.5, ls="", mfc="white", mec="k",
            label="exact expectation at that mean")
    ax.axhline(entry["alc"], color=C["legacy"], lw=0.7, ls="--")
    ax.plot(entry["q_if_empirical_mean"], entry["alc"], marker="v", ms=4, color=C["legacy"],
            ls="", label=f"organisers' entry {entry['alc']:g}, at the E[p(1−p)] an\n"
            f"empirical mean would need ({entry['q_if_empirical_mean']:.3f})")
    ax.set_xlabel("the run's E[p(1−p)] over its pairs")
    ax.set_ylabel("empirical-mean ALC of the run")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.4, labelspacing=0.9)
    fig.subplots_adjust(left=0.09, right=0.6, top=0.97, bottom=0.17)
    return fig


# ---------------------------------------------------------------------------
# the item-level gap nobody reaches

IG_REGIMES = [("tl", "tuned test-like"), ("tl mix/whole", "mix/whole"),
              ("r1b", "public R1,\nbenchmark-first"), ("r1p", "public R1,\npair-uniform")]


def fig_item_gap(inp):
    s = inp.load("results/itemsig_eval.json")["summary"]["regimes"]
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.62, 2.45))
    w = 0.25
    bars = [("pair_rate_oracle", "pair-rate oracle (the pair's own rate)", "#bbbbbb"),
            ("item_oracle", "item oracle (in-sample Rasch difficulty)", "#555555"),
            ("base_b31", "shipped hier at B31", C["ship"])]
    for k, (key, _) in enumerate(IG_REGIMES):
        o = s[key]["oracle"]
        for m, (fld, lab, col) in enumerate(bars):
            ax.bar(k + (m - 1) * w, o[fld], width=w * 0.92, color=col,
                   label=lab if k == 0 else None)
        g = o["gap"]
        b31 = s[key]["nested_within"]["budgets"][-1]["diff"]
        ax.annotate(f"gap {g['mean']:.3f} ± {g['cluster_se']:.3f}\n"
                    f"itemsig at B31: {-b31 / g['mean']:.1%}",
                    (k, max(o["pair_rate_oracle"], o["base_b31"]) + 0.004), ha="center",
                    va="bottom", fontsize=4.9)
    ax.set_xticks(range(len(IG_REGIMES)))
    ax.set_xticklabels([lab for _, lab in IG_REGIMES], fontsize=5.8)
    ax.set_ylabel("Brier on the evaluated responses")
    ax.set_ylim(0, 0.265)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=5.1, ncol=1, borderaxespad=0.1)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)
    return fig


# ---------------------------------------------------------------------------
# the empirical-Bayes level centre by budget

EB_REGIMES = [("r1b", "public, benchmark-first"), ("r1p", "public, pair-uniform"),
              ("tl no shift", "test-like, no date shift"),
              ("tl lm-1.2", "test-like, level_mean −1.2"), ("tl", "test-like, default"),
              ("tl lm-2.0", "test-like, level_mean −2.0")]
EB_CONFIG = "hierEB-cs,tm=2.0@hier"


def fig_eb_adaptation(inp):
    lc = inp.load("results/level_calibration.json")["summary"]["eb_adaptation"]
    rs = inp.load("results/regime_sensitivity.json")["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.0), sharey=True)
    x = bx()
    cmap = plt.get_cmap("cividis")
    levels = [lc["pair_logit_mean"][k] for k, _ in EB_REGIMES]
    levels += [rs["realised"][r]["mean"] for r in rs["realised"]]
    lo, hi = min(levels), max(levels)

    def colour(level):  # one level scale for every panel: dark = low
        return cmap(0.05 + 0.85 * (level - lo) / (hi - lo))

    ax = axes[0]
    handles = []
    marks = "osD^vPX"
    for n, (key, lab) in enumerate(sorted(EB_REGIMES, key=lambda kv: lc["pair_logit_mean"][kv[0]])):
        e = lc["estimates"][f"{EB_CONFIG} | {key}"]
        lev = lc["pair_logit_mean"][key]
        ls = "--" if key.startswith("r1") else "-"
        ax.plot(x, [e[f"B{b}"][0] for b in BUDGETS], color=colour(lev), marker=marks[n],
                ms=2.4, ls=ls, lw=1.0)
        handles.append((Line2D([], [], color=colour(lev), ls=ls, marker=marks[n], ms=2.4),
                        f"{lab} ({lev:+.2f})"))
    budget_axis(ax, label=False)
    ax.set_ylabel("EB level centre (item-level logit)")
    title(ax, "a", "EB on hier's defaults", fontsize=6.2)
    fig.legend([h for h, _ in handles], [lab for _, lab in handles], loc="upper left",
               bbox_to_anchor=(0.06, 0.255), fontsize=5.0, title="level calibration regimes "
               "(mean pair logit)", title_fontsize=5.2, labelspacing=0.25, ncol=1)
    order = sorted(rs["realised"], key=lambda r: rs["realised"][r]["mean"])
    handles = []
    for j, cfg in enumerate(("eb_ship", "eb_adapt")):
        ax = axes[1 + j]
        for n, reg in enumerate(order):
            tr = rs["eb_traces"].get(f"{cfg} | {reg}")
            if not tr:
                continue
            lev = rs["realised"][reg]["mean"]
            ls = "--" if reg.startswith("R1") else "-"
            ax.plot(x, [tr[f"B{b}"]["mu0"] for b in BUDGETS], color=colour(lev),
                    marker=marks[n], ms=2.3, ls=ls, lw=0.9)
            if j == 0:
                handles.append((Line2D([], [], color=colour(lev), ls=ls, marker=marks[n],
                                       ms=2.3), f"{reg} ({lev:+.2f})"))
        budget_axis(ax, label=False)
        title(ax, "bc"[j], f"{NAMES[cfg]} (P1a)", fontsize=6.2)
    fig.legend([h for h, _ in handles], [lab for _, lab in handles], loc="upper left",
               bbox_to_anchor=(0.43, 0.255), fontsize=5.0, title="P1a regimes (realised mean "
               "pair logit)", title_fontsize=5.2, labelspacing=0.25, ncol=2,
               columnspacing=1.0)
    fig.text(0.53, 0.29, BUDGET_LABEL, ha="center", fontsize=6.4)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.39, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# item difficulty does not transfer

TR_ROWS = [  # label, block, key (None: the block is per benchmark)
    ("TF-IDF+SVD ridge, leave one benchmark out", "lobo", "tfidf_ridge_centred"),
    ("embedding ridge, leave one benchmark out", "lobo", "emb_ridge_centred"),
    ("embedding kNN, leave one benchmark out", "lobo", "emb_knn_centred"),
    ("embedding ridge, within benchmark (5-fold)", "within", "emb_ridge"),
    ("item_features group mean, within benchmark", "group_only", None),
    ("embedding ridge, whole groups held out", "within_groupcv", "emb_ridge"),
]


def fig_transfer(inp):
    e = inp.load("results/emb_transfer.json")
    h = inp.load("results/harness_thresholds.json")
    lo = h["thresholds"]["honest"]["transferred nested"]["smallest_r_majority_of_draws"]
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, 2.5))
    n = len(TR_ROWS)
    w = 0.8 / n
    greys = ["#000000", "#E69F00", "#F5C46B", "#0072B2", "#56B4E9", "#999999"]
    for k, (lab, block, key) in enumerate(TR_ROWS):
        vals, errs = [], []
        for p in PARENTS:
            u = e[block][key][p] if key else e[block][p]
            vals.append(u["pearson"])
            ci = u.get("pearson_ci_groupboot")
            errs.append([u["pearson"] - ci[0], ci[1] - u["pearson"]] if ci else [0, 0])
        pos = np.arange(len(PARENTS)) + (k - (n - 1) / 2) * w
        err = np.asarray(errs).T
        ax.bar(pos, vals, width=w * 0.9, color=greys[k], label=lab,
               yerr=err if err.any() else None,
               error_kw={"elinewidth": 0.6, "capsize": 0, "ecolor": "#777777"})
    ax.axhline(lo, color="k", lw=0.7, ls=(0, (5, 2)))
    ax.text(len(PARENTS) - 0.5, lo + 0.01, f"honest r {lo:g}: the gate's bar", fontsize=5.4,
            ha="right", va="bottom")
    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_xticks(range(len(PARENTS)))
    ax.set_xticklabels(PARENTS)
    ax.set_ylabel("Pearson r with item difficulty")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=5.2, ncol=2, bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(-0.5, 0.95)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.97, bottom=0.1)
    return fig


# ---------------------------------------------------------------------------
# hier's ablations and sensitivities

ABLATIONS = [  # option in results/hier_eval.json, label
    ("hier -attributes", "attribute prior off"),
    ("hier -pool_mu", "level pooling off"),
    ("hier -delta", "pair deviation off"),
    ("hier -groups", "feature groups off"),
    ("hier -link", "linking off"),
    ("hier relink 0.1", "relink 0.1"),
    ("hier relink 0.3", "relink 0.3"),
    ("hier t3 level", "Student-t level (nu 3)"),
    ("hier -line", "line off (Laplace)"),
    ("hier guess 1 (hard floor)", "hard floor (guess 1)"),
    ("hier centre zero", "level centre 0"),
    ("hier +text", "text term on"),
    ("hier sigma_mu x0.5", "sigma_mu ×0.5"),
    ("hier sigma_mu x2", "sigma_mu ×2"),
    ("hier sigma_delta x0.5", "sigma_delta ×0.5"),
    ("hier sigma_delta x2", "sigma_delta ×2"),
]


def bonferroni_z(k: int, alpha: float = 0.05) -> float:
    """Two-sided normal critical value for k comparisons."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / (2 * k))


def fig_hier_ablations(inp):
    h = inp.load("results/hier_eval.json")
    setting = "benchmark/pair"
    dd = h["r1"][setting]["diffs"]
    k = h["config"]["n_comparisons"]
    z = bonferroni_z(k)
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH * 0.75, 3.2))
    ys = np.arange(len(ABLATIONS))
    for y, (key, _) in zip(ys, ABLATIONS):
        v = dd[key]["hier"]
        d, se, ss = v["diff"], v["cluster_se"], v["cluster_se_strat"]
        ax.plot([d - z * se, d + z * se], [y, y], color=C["ship"], lw=0.6)
        ax.plot([d - se, d + se], [y, y], color=C["ship"], lw=2.2, alpha=0.8,
                solid_capstyle="butt")
        ax.plot([d - z * ss, d + z * ss], [y + 0.3, y + 0.3], color=C["light"], lw=0.6)
        ax.plot(d, y, marker="o", ms=3.0, color=C["ship"], mec="white", mew=0.3, zorder=4)
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab} ({dd[key]['hier']['runs']})" for key, lab in ABLATIONS],
                       fontsize=5.8)
    ax.set_ylim(len(ABLATIONS) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    milli(ax, "x")
    ax.set_xlabel("ALC: option − hier's default (10$^{-3}$; public R1, benchmark-first)")
    handles = [(Line2D([], [], color=C["ship"], lw=2.2), "±1 pair-cluster SE"),
               (Line2D([], [], color=C["ship"], lw=0.6),
                f"±{z:.3f} pair-cluster SE (Bonferroni, {k} comparisons)"),
               (Line2D([], [], color=C["light"], lw=0.6), f"±{z:.3f} stratified SE")]
    ax.legend([h_ for h_, _ in handles], [lab for _, lab in handles], loc="upper center",
              bbox_to_anchor=(0.4, -0.13), fontsize=5.3, ncol=2, columnspacing=1.0)
    fig.subplots_adjust(left=0.3, right=0.98, top=0.98, bottom=0.24)
    return fig


# ---------------------------------------------------------------------------
# every idea against the gate

def _harness_line(block, line):
    ln = block["lines"][line]
    if "regimes" in ln:  # itemcov's layout
        rg = ln["regimes"]
        return (rg["tl"]["est"], rg["tl"]["cluster_se"],
                max(rg["r1b"]["est"], rg["r1p"]["est"]))
    return ln["tl"], ln.get("tl_cluster_se", 0.0), max(ln["r1b"], ln["r1p"])


def ideas_rows(inp):
    """(group, label, kind, test-like difference, its cluster SE, worse public
    weighting or None); kind is 'nested' (the gate's line), 'forced' or 'fixed'."""
    rows = []
    it = inp.load("results/itemsig_eval.json")["summary"]["regimes"]

    def isig(key):
        v = it["tl"][key]
        a = v.get("ALC", v)
        pub = [it[r][key].get("ALC", it[r][key])["mean"] for r in ("r1b", "r1p")]
        return a["mean"], a["cluster_se"], max(pub)

    g = "item-side layer (itemsig)"
    rows += [(g, "nested, selected within regime", "nested", *isig("nested_within")),
             (g, "nested, joint selection", "nested", *isig("nested_joint")),
             (g, "library default", "fixed", *isig("default")),
             (g, "best in sample", "fixed", *isig("in_sample_best"))]
    ic = inp.load("results/itemcov_eval.json")["harness"]
    g = "known-sign item cues"
    for cue in ("stated_size", "position", "position_within", "format_score", "log_length",
                "image_ref"):
        rows.append((g, f"{cue}, nested", "nested", *_harness_line(ic[cue], "allowed nested")))
    rows.append((g, "stated_size, per-pair s 0.5 from B7", "forced",
                 *_harness_line(ic["stated_size"], "per-pair s=0.5 from B7 (forced)")))
    ss = inp.load("results/subject_side.json")["summary"]
    g = "subject side"
    nest = ss["nested"]["primary"]
    rows.append((g, "combined selection, nested", "nested", nest["tl"]["mean"] if "mean" in
                 nest["tl"] else nest["tl"]["ALC"]["mean"], nest["tl"]["ALC"]["cluster_se"]
                 if "ALC" in nest["tl"] else nest["tl"]["cluster_se"],
                 max(nest[r]["ALC"]["mean"] if "ALC" in nest[r] else nest[r]["mean"]
                     for r in ("r1b", "r1p"))))
    for c, lab in (("E", "E, ordered reasoning effort"), ("H", "H, harness identity"),
                   ("Dlog", "Dlog, log release date"), ("Dclip", "Dclip, clipped release date"),
                   ("H+E", "H+E")):
        a = ss["per_config"][c]["ALC"]
        rows.append((g, f"{lab}, nested", "nested", a["tl"]["mean"], a["tl"]["cluster_se"],
                     max(a["r1b"]["mean"], a["r1p"]["mean"])))
    st = ss["student_t"]["regimes"]
    for c in ("T", "T1.8"):
        d_ = st["tl"]["configs"][c]["dalc"]
        rows.append((g, f"Student-t level {c}", "fixed", d_["mean"], d_["cluster_se"],
                     max(st[r]["configs"][c]["dalc"]["mean"] for r in ("r1b", "r1p"))))
    he = inp.load("results/heads_eval.json")["rows"]["current"]["configs"]
    g = "meta-learned heads"
    for c, kind in (("lopo_diff", "nested"), ("lopo_kern", "nested"), ("lopo_ass", "nested"),
                    ("lopo_all", "nested"), ("force_all_0.01", "forced"),
                    ("force_all_0.1", "forced"), ("force_all_1.0", "forced")):
        v = he[c]["tl"]
        rows.append((g, c.replace("_", " "), kind, v["alc"], v["cluster_se"], None))
    g = "language-model and encoder probes"
    probes = [
        ("results/llm4b_close.json", ("harness", "rating"), "4B judge rating"),
        ("results/hidden_state_probe.json", ("harness", "entropy"), "4B entropy head"),
        ("results/hidden_state_probe.json", ("harness", "hidden"), "4B hidden-state head"),
        ("results/finetune_encoder.json", ("harness", "frozen_lobo512"), "frozen embedding head"),
        ("results/finetune_encoder.json", ("harness", "finetuned"), "fine-tuned encoder"),
        ("results/strong_llm_eval.json", ("harness", "head rubric_ridge"), "14B rubric head"),
        ("results/strong_llm_eval.json", ("harness", "head judge_ridge"), "14B judge head"),
        ("results/strong_llm_eval.json", ("harness", "time_log_minutes"), "14B time_log_minutes"),
        ("results/strong_llm_eval.json", ("entropy", "harness", "ent_first1024"),
         "14B entropy (commit D)"),
    ]
    for f, path, lab in probes:
        b = inp.load(f)
        for p_ in path:
            b = b[p_]
        rows.append((g, f"{lab}, nested", "nested", *_harness_line(b, "transferred nested")))
        rows.append((g, f"{lab}, transferred from B1", "forced",
                     *_harness_line(b, "transferred from B1 (forced)")))
    mq = inp.load("results/mcq_floor.json")["regimes"]
    v = {r: mq[r]["compare"]["fix - ship"]["all"]["per_appearance"] for r in mq}
    rows.append(("adopted, for contrast", "multiple-choice floor corrected", "fixed",
                 v["tl"]["est"], v["tl"]["cluster_se"], max(v["r1b"]["est"], v["r1p"]["est"])))
    return rows


IDEAS_XLIM = (-0.0032, 0.0022)


def fig_ideas_forest(inp):
    rows = ideas_rows(inp)
    gate = inp.load("results/harness_thresholds.json")["meta"]["gate"]["tl"]
    groups = list(dict.fromkeys(r[0] for r in rows))
    ypos, labels, y = [], [], 0
    heads = []
    for g in groups:
        heads.append((y, g))
        y += 1
        for r in rows:
            if r[0] == g:
                ypos.append(y)
                labels.append(r[1])
                y += 1
        y += 0.4
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, 6.8))
    lo, hi = IDEAS_XLIM
    for yy, r in zip(ypos, rows):
        _, _, kind, d, se, pub = r
        col = C["ship"] if kind == "nested" else C["other"]
        filled = kind == "nested"
        if lo < d < hi:
            ax.plot([max(d - se, lo), min(d + se, hi)], [yy, yy], color=col, lw=1.6,
                    solid_capstyle="butt", alpha=0.85)
            ax.plot(d, yy, marker="o", ms=3.0, color=col, mfc=col if filled else "white",
                    mec=col, mew=0.7, zorder=4)
        else:
            edge, sgn = (hi, 1) if d >= hi else (lo, -1)
            ax.annotate("", xy=(edge, yy), xytext=(edge - 0.0004 * sgn, yy),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=col))
            ax.text(edge - 0.00045 * sgn, yy, f"{d * 1e3:+.1f}", fontsize=4.8, va="center",
                    ha="right" if sgn > 0 else "left", color=col)
        if pub is not None and lo < pub < hi:
            ax.plot(pub, yy + 0.28, marker="^", ms=2.6, ls="", color=C["legacy"], mew=0)
        elif pub is not None:
            edge, sgn = (hi, 1) if pub >= hi else (lo, -1)
            ax.plot(edge, yy + 0.28, marker=">" if sgn > 0 else "<", ms=2.6, ls="",
                    color=C["legacy"], mew=0, clip_on=False)
            ax.text(edge - 0.0001 * sgn, yy + 0.28, f"{pub * 1e3:+.1f}", fontsize=4.6,
                    va="center", ha="right" if sgn > 0 else "left", color=C["legacy"])
    for yy, g in heads:
        ax.text(-0.5, yy, g, fontsize=6.0, fontweight="bold", va="center", ha="left",
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#888888", lw=0.6)
    ax.axvline(gate, color="k", lw=0.8, ls=(0, (5, 2)))
    ax.text(gate, -1.0, f"gate {gate * 1e3:g}", fontsize=5.6, ha="center", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=5.4)
    ax.set_ylim(y - 0.2, -1.2)
    ax.set_xlim(lo, hi)
    ax.grid(axis="y", visible=False)
    milli(ax, "x")
    ax.set_xlabel("test-like ALC difference vs the shipped model (10$^{-3}$; negative: better)")
    handles = [(Line2D([], [], color=C["ship"], marker="o", ms=3, lw=1.6),
                "nested line, ±1 cluster SE (what the gate reads)"),
               (Line2D([], [], color=C["other"], marker="o", ms=3, mfc="white", lw=1.6),
                "forced or fixed configuration, ±1 cluster SE"),
               (Line2D([], [], color=C["legacy"], marker="^", ms=2.6, ls=""),
                "worse public weighting (◂: off scale, value)")]
    ax.legend([h_ for h_, _ in handles], [lab for _, lab in handles], loc="upper left",
              bbox_to_anchor=(-0.5, -0.06), ncol=3, fontsize=5.0, columnspacing=0.8,
              handletextpad=0.3)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.975, bottom=0.09)
    return fig


# ---------------------------------------------------------------------------
# registry and output

FIGURES = [
    ("learning_curves", fig_learning_curves),
    ("level_surface", fig_level_surface),
    ("gate_curve", fig_gate_curve),
    ("regime_sensitivity", fig_regime_sensitivity),
    ("formative_runs", fig_formative_runs),
    ("level_fix_budgets", fig_level_fix_budgets),
    ("eb_adaptation", fig_eb_adaptation),
    ("item_gap", fig_item_gap),
    ("transfer", fig_transfer),
    ("empirical_mean", fig_empirical_mean),
    ("hier_ablations", fig_hier_ablations),
    ("ideas_forest", fig_ideas_forest),
]


def save(fig, out: Path, name: str) -> dict[str, str]:
    svg, png = out / f"{name}.svg", out / f"{name}.png"
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None})
    fig.savefig(png, format="png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return {p.name: sha256_file(p) for p in (svg, png)}


def build(out: Path = OUT, only=None, root: Path = ROOT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inp = Inputs(root)
    mp = out / "manifest.json"
    old = json.loads(mp.read_text()).get("figures", {}) if (only and mp.exists()) else {}
    figs = {}
    with plt.rc_context(RC):
        for name, fn in FIGURES:
            if only and name not in only:
                if name in old:
                    figs[name] = old[name]
                continue
            inp.begin()
            files = save(fn(inp), out, name)
            figs[name] = {"function": fn.__name__, "inputs": inp.record(), "outputs": files}
    manifest = {
        "script": SCRIPT,
        "script_sha256": sha256_file(root / SCRIPT),
        "command": "python tools/report_figures.py",
        "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__},
        "settings": {"backend": "Agg", "png_dpi": DPI, "svg_fonttype": RC["svg.fonttype"],
                     "svg_hashsalt": RC["svg.hashsalt"], "font": RC["font.family"],
                     "width_in": WIDTH},
        "figures": figs,
    }
    mp.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def check(out: Path = OUT, root: Path = ROOT, rebuild: bool = True) -> list[str]:
    """What is wrong with the drawn figures: a missing manifest or file, an
    input or the script changed since drawing, an input outside results/, a
    figure that no longer builds or reads other files, or (same matplotlib) a
    fresh build that differs from the committed bytes."""
    mp = out / "manifest.json"
    if not mp.exists():
        return [f"{mp} missing"]
    man = json.loads(mp.read_text())
    problems = []
    if man.get("script_sha256") != sha256_file(root / SCRIPT):
        problems.append("the script changed since the figures were drawn")
    for name, _ in FIGURES:
        m = man["figures"].get(name)
        if m is None:
            problems.append(f"{name}: not in the manifest")
            continue
        for rel, sha in m["inputs"].items():
            if not Inputs.allowed(rel):
                problems.append(f"{name}: input {rel} is not results/*.json")
            elif not (root / rel).exists() or sha256_file(root / rel) != sha:
                problems.append(f"{name}: {rel} changed since the figure was drawn")
        for fn, sha in m["outputs"].items():
            p = out / fn
            if not p.exists():
                problems.append(f"{name}: {fn} missing")
            elif sha256_file(p) != sha:
                problems.append(f"{name}: {fn} differs from the manifest")
    if rebuild:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = build(Path(tmp), root=root)
        same = fresh["versions"]["matplotlib"] == man["versions"]["matplotlib"]
        for name, f in fresh["figures"].items():
            m = man["figures"].get(name)
            if m is None:
                continue
            if f["inputs"] != m["inputs"]:
                problems.append(f"{name}: reads {sorted(f['inputs'])}, the manifest says "
                                f"{sorted(m['inputs'])}")
            if same and f["outputs"] != m["outputs"]:
                problems.append(f"{name}: a fresh build differs from the committed files")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", choices=[n for n, _ in FIGURES])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        probs = check(a.out)
        for p in probs:
            print("PROBLEM", p)
        print("ok" if not probs else f"{len(probs)} problem(s)")
        return 1 if probs else 0
    man = build(a.out, a.only)
    for name, f in man["figures"].items():
        print(f"{name:20s} {', '.join(f['outputs'])}  <- {', '.join(f['inputs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
