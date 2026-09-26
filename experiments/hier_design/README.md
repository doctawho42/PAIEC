# Design scripts behind `paiec/hier.py` and `paiec/prior.py` (provenance)

These are the step-2 scripts that set the structure and hyperparameters of the
hierarchical model, together with the checks from its reviews and from the audit
of its evaluation. They were written in a scratch directory while the model was
being built. They are copied here so that the numbers `docs/findings.md`
("Hierarchical model") and the docstrings of `paiec/hier.py` and
`paiec/prior.py` cite have a script in the repository. The only change is to
paths: every script finds the repository and its sibling modules relative to
its own location.

They are provenance, not maintained experiments. They read `paiec` internals
(`_Fit`, `LINE_MAX`, `Problem`, ...) as they were at the time, and their
recorded outputs sit next to them. The subdirectories keep the scratch layout,
so a citation such as `final_verify/c1_calib.py` or `final_fix/v2_after.txt`
resolves here. Where a script writes a file, it writes it next to itself;
`.gitignore` keeps the caches and bulky outputs out of git.

"Rerun" below means that this copy was rerun, in a scratch export, against the
library of commit `bba726c`, the commit whose code produced
`results/hier_eval.json`. During this work `paiec/hier.py` and `paiec/prior.py`
were being changed by another task, so the reruns used a clean export of that
commit, not the working tree. The recorded outputs kept here are the originals,
with library warning lines and scratch paths stripped. Where a rerun is marked
identical, it reproduced them line for line or value for value. A later library will not reproduce these numbers exactly. That holds
for the `Hyper` defaults above all: with the change in progress, for example,
the no-attribute link weight in `v3b_after.txt` moved from 0.16 to 0.18.

## Level prior, group share, identity (the three step-2 analyses)

Run from this directory, in this order: `runs.py` writes `runs.pkl`, which the
next three read; `levels.py` writes `levels.json`, which `prior_sweep.py` and
`predictor_check.py` read.

| script | produces | cited for | status |
|---|---|---|---|
| `runs.py` | `runs.pkl` (gitignored): the 600 R1 runs of `official_baselines.py`, seed 0, as labels and evaluation rates | the input of the exact count-based scorer (`score.py`) | rerun (100 s on six processes); its runs feed the reruns below |
| `levels.py` | `levels.json` | public levels: mean -0.56, sd 0.90, pairs within sd 0.93, total pair-logit sd 1.29 | rerun: `levels.json` identical |
| `prior_sweep.py` | `prior_sweep_out.txt` (stdout), `prior_sweep.json` (gitignored) | level prior widths (`prior.WIDEN`), the regret table | rerun: stdout identical to `prior_sweep_out.txt` except its trailing 'done' line (17 minutes at a load average of 200) |
| `cluster_se.py` | `cluster_se.json` | pair-cluster SEs of the level-prior candidates | rerun: `cluster_se.json` identical (22 minutes at a load average of 200) |
| `predictor_check.py --runs 300 --jobs 6` | `predictor_check_out.txt` (stdout), `predictor_check.json` (gitignored) | a level centre added to the shipped Predictor (+0.0006 LOBO, +0.0031 in-sample) | not rerun (300 replica runs of six Predictor variants) |
| `identity_prior.py` | `identity_prior_out.txt` (stdout; the scratch copy was `out2.txt`) | identity does not transfer beyond attributes; tau2_res -0.06 [-0.21, 0.07] | rerun: stdout identical to `identity_prior_out.txt` (3 minutes) |
| `item_signal.py` | `item_signal_out.txt` (stdout), `item_signal.json` (gitignored) | item_features variance shares and group-effect gains; `prior.G_CAP` | not rerun (about 10 minutes) |

`common.py` and `score.py` are shared helpers. `common.flat()` caches every
eligible response in `flat.parquet` (gitignored).

The level-width sweep scored the same R1 runs (seed 0, runs 0 to 599) that
`experiments/hier_eval.py` evaluates on its primary setting. So the widths were
not chosen on held-out runs. The bias runs against hier, though: the widened
prior chosen there scores 0.2161 on those runs against the best 0.2102
(`prior_sweep_out.txt`, 'R1 exact').

The identity estimate (tau2_res) comes from attribute residuals fitted
leave-one-benchmark-out, and a model's rows on other benchmarks stay in that
fit. The final review of step 2 found this estimator biased downward (a name's
residuals on two benchmarks are pulled apart). A change to `paiec/prior.py` for
it was in progress during this work, so read the identity conclusion as
provisional.

## Final review of the model (`final_verify/`)

Exact posterior predictives by quadrature or HMC, written from the model
statement, against `HierPredictor`. `common.py`, `sim.py`, `exact1.py`,
`v2_bench.py` and `v3_link.py` are the shared pieces. Run a script from inside
`final_verify/`, or by its path, since the sibling imports resolve from the
script's directory.

| script | produces | cited in | status |
|---|---|---|---|
| `v2_bench.py gauss gauss+floor` | stdout (recorded as `final_fix/v2_after.txt`, the rerun after the fixes) | hier.py: the point cases (0.365 against the exact 0.295; new subject after three 1/7) | rerun: identical to `final_fix/v2_after.txt` |
| `v3b_link_t.py` | stdout (recorded as `final_fix/v3b_after.txt`) | hier.py: linked Student-t levels, off by up to 0.010 without attributes, 0.025 at weight 0.3 | rerun: identical to `final_fix/v3b_after.txt` |
| `d3_exact_dense.py n_s reps nu` | `d3_2_g.txt`, `d3_2_t.txt`, `d3_26_g.txt`, `d3_26_t.txt` (n_s 2 or 26, Gaussian or t level) | hier.py: the line's bias toward 0.5 with many subjects (0.014 at B1, 0.007 at B3, 0.003 from B7) | not rerun (26 subjects take about 10 minutes a rep); provenance |
| `d4_star_sim.py m B reps [floor]` | stdout, not kept | hier.py: labeled-item targets, line 0.004 to 0.012 against Laplace 0.005 to 0.014 | output not kept; provenance until rerun |
| `d1_linemax.py [n_s] [scope]` | stdout, not kept | hier.py: LINE_MAX cap against no cap on dense runs | output not kept; provenance until rerun |
| `c1_calib.py KIND runs jobs` (KIND formative or dense) | `c1_dense.txt` (dense, 30 runs); the formative output was lost (the recorded attempt crashed) | prior.py: REFERENCE sigma_delta 1.0 cost 0.0021 ALC on runs simulated from the default model | not rerun (240 simulated formative runs of seven hyperparameter sets, 30 to 60 minutes at the load during this work); the 0.0021 is provenance |

## Fixes after the review (`final_fix/`)

| script | produces | cited in | status |
|---|---|---|---|
| `c1_after.py formative runs jobs` | `c1_after.txt` (240 runs) | prior.py: mu0 = -4.14 left by a strict run-LOBO fit cost 0.013 ALC ('old: multi_swebench left', +0.0138); old against new REFERENCE | not rerun; provenance |
| `d2_after.py n_s reps seed`, `d2b.py n_s reps seed n_items feats` | `d2_after.txt`, `d2b.txt`, `d2c.txt` | hier.py: the old all-or-nothing LINE_MAX cap (0.0052 off where the others were not) | cannot be rerun as recorded: they switch `paiec.hier.LINE_CORE`, which the library no longer has, so the old-cap variant would silently equal the new one; provenance |
| `plane.py`, `plane_star.py [gauss] [gauss+floor]` | stdout, not kept | hier.py: a 2-D grid over (a'x, J'x) moves the labeled-item corner case only from +0.070 to +0.061 | a monkeypatched prototype of `_Fit._components`; provenance |

## Student-t quadrature (`fix3/`)

`lam_quad.py` compares quadratures over the scale-mixture weight of a
Student-t level. It is cited in hier.py for "16 Laguerre nodes were 8e-2 off".
Rerun: `lam_quad_out.txt`, where gl16 is 8.09e-02 off, as cited.

## Audit of the step-2 evaluation (`audit/`)

These two checks cannot be computed from the rows in `results/hier_eval.json`:
one needs the predictions themselves, the other refits. Both import
`experiments/hier_eval.py`.

| script | produces | cited in findings.md | status |
|---|---|---|---|
| `null_ece.py` | `null_ece_out.txt` (stdout), `null_ece.json` | hier's ECE-ALC excess over the Predictor (+0.0108) is mostly the binned estimator's floor (+0.0095 +- 0.0006); excess +0.0014 +- 0.0022 | rerun: identical (same per-run values; about 3 minutes on four processes) |
| `strict_ref.py` (env `SD`, default 1.0) | `strict_ref_out.txt` (stdout), `strict_ref_1.0.json` | strict run-LOBO with REFERENCE sigma_delta 1.0 instead of 2.5 | rerun: stdout and `strict_ref_1.0.json` identical (5 minutes on three processes) |

The audit's other statistics (a bootstrap stratified by benchmark, results
without swe_rebench, the breakdown by pairs of a benchmark per run) are now part
of `experiments/hier_eval.py --summarise`.
