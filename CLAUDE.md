# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the Predictive AI Evaluation Competition (PAIEC, NeurIPS 2026): predict whether an AI system (subject) answered a benchmark item correctly, given 0–31 acquired labels, scored by Brier ALC. The test is a benchmark-level cold start: hidden benchmarks never appear in the public data. The repo holds a replica of the official streaming evaluator, the predictors measured against it, and one experiment script per reported number. `docs/protocol.md` specifies the verified official protocol and what is still unknown; `docs/findings.md` maps every number to the script that produces it (the section "Under the official protocol" supersedes the older, legacy-replica numbers, including those in `README.md`).

## Commands

```bash
pip install -e ".[dev]"
python -m paiec.fetch                  # download gated measurement-db into data/ (needs accepted HF terms + HF_TOKEN)
pytest                                 # protocol invariants; run from repo root, uses synthetic data, no download needed
pytest tests/test_official.py::<name>  # single test
python experiments/official_baselines.py --runs 600 --jobs 6   # baselines under the official protocol (~12 min; --no-dense skips R2)
python tools/build_submission.py       # fit prior, build dist/paiec.zip, run every packaging check + the organisers' validator
python third_party/paiec_baseline/check_submission_zip.py dist/paiec.zip   # the validator on its own
```

Data lives in `data/<benchmark>/{response,items,subjects}.parquet` (gitignored); override the location with `PAIEC_DATA`. `pytest` runs without the data (tests use synthetic pairs; `tests/test_submission.py` uses `submission/prior.json` if a build wrote one, else a stand-in); everything else needs the data. `third_party/paiec_baseline/` is a gitignored copy of github.com/aims-foundations/paiec_baseline (the official validator, streaming client `tools/streaming_ingestion.py`, reference predictors) — re-clone it if missing; it has no license, so never commit it.

## Architecture

**Official replica (`paiec/official.py`)** is the ground truth for all new measurements; it mirrors the organisers' `streaming_ingestion.py`. Scoring is budget-major: every sampled pair is evaluated at budget 0 (empty `labeled`), acquisition continues until each pair has up to the next budget's labels, then every pair is evaluated again, for 0,1,3,7,15,31. At a checkpoint one shared `labeled` list (every sampled pair's first B labels — other subjects and other benchmarks included) goes to every target. `run_official(run, model_factory, ...)` calls the factory afresh per checkpoint (and per simulated worker), because the platform recreates evaluation workers, so no in-memory state survives between budgets. Inputs are official-format only (8-key subject, 4-key item, anonymous `benchmark_id`, no private ids). Default acquisition is the platform's exact sha256 rule. `sample_run` draws formative-like runs (≤1,000 subject-item pairs, 5–12 pairs); `dense_run` takes every pair of one benchmark. Unknowns are parameters, not assumptions — notably `split_scope` ('pair' vs 'benchmark'), which decides whether other subjects' labels land on a target's own evaluation items. Pass the run's benchmarks to prior fitting via `training_pairs(pairs, run)` to stay leave-one-benchmark-out. `tests/test_official.py` pins the invariants.

`paiec/evaluator.py` is the **legacy** pair-major replica with the wrong `labeled` semantics; keep it only so older experiments and `tests/test_evaluator.py` still reproduce. Its numbers (0.181, 0.1725, 0.1898, the ladder) are not comparable with the official ones. Both use `stable_hash`, never Python `hash()` (salted per process).

**Two predictor paths that share modules but are used differently:**
- `paiec/pipeline.py` — offline, pooled scoring used by experiments: `build(pairs)` precomputes item difficulty/embeddings, then `score(...)` evaluates over precomputed trajectories (`paiec/trajectories.py`). Fast enough for multi-seed sweeps.
- `paiec/predict.py` — the run-time `Predictor` the submission uses. Model: `p = c + (1-c-slip) * sigmoid(kappa*(a + b*z))`, where `c` is the MCQ guessing floor (`mcq.py`), `z` item difficulty from a joint IRT fitted on visible labels with text embeddings as prior (`irt.py`, only once a benchmark has `BenchmarkFit.WARMUP`=64 distinct labeled items), `a` subject ability with a relative prior from subject attributes (`subjects.py`, coefficients from `fit_prior`), `b` per-pair slope, and `kappa` Laplace shrinkage from `fitting.fit_ab`. `predict` must stay a **pure function of (input, labeled)**: fits are cached under a content fingerprint of `labeled`, items/subjects are keyed on digests of all their visible fields (the official input has no ids; text prefixes collide), and any failure returns a finite fallback instead of raising. Known gap: the benchmark's level in other subjects' labels is not used (see findings, dense multi_swebench).

**Submission (`submission/model.py`)** is the competition entry point, `predict(input, labeled)`. It ships `paiec.hier.HierPredictor` with the level prior moved down for the hidden test (`LEVEL` in model.py: mu0 -2.5, sigma_mu 2.5, attr_scale 0.5; docs/findings.md, "Calibrating for the hidden test" → "What actually shipped, after the audit"). `prior.json` is the `paiec.prior.to_json` bundle of `prior.build` on every eligible public pair with `LEVEL` written over it; the build reads `LEVEL` from model.py, and model.py uses it alone if prior.json is unusable. A legacy `{coef, spec}` prior.json (`build_submission.py --legacy`) loads the old Predictor instead, as a rollback. `model.make()` gives a fresh predictor, as each recreated worker holds one. The archive ships `paiec/{__init__,hier,prior,predict,fitting,irt,subjects,mcq}.py` renamed to `paiec_rt/` (so no platform-side `paiec` can shadow it), imported at module level (the validator restores `sys.path` right after loading, so lazy imports fail silently); `prior.py` ships whole, its `main()` offline only. Shipped modules may import only the standard library, numpy and each other at module level; hier needs nothing else (scipy/sklearn only for the legacy Predictor, lazily with numpy fallbacks; pandas only to parse a non-ISO release date). `tools/build_submission.py` enforces all of this, checks that the archive predicts bit for bit like the in-repo predictor on the same prior.json, and runs the organisers' validator; `dist/paiec.zip` exists only if every check passed. `labeling.py` is excluded by default because no policy beat the platform's random one.

**Research-only modules** (not shipped): `paiec/testlike.py` builds test-like runs (pseudo-benchmarks, ~one pair per benchmark, tuned to the real formative feedback) and is the primary regime for choices aimed at the hidden test; `paiec/itemsig.py` (similarity-based residual layer) measured null and is not shipped; `paiec/llmfeat.py` + `experiments/llm_features.py` extract local-LLM item features into gitignored `data/features/`.

**Experiments** — one script per result in `experiments/`; `_*_legacy.py` are earlier versions kept for reproducibility of older numbers. `experiments/llm_rating/` is the LLM-judged difficulty study; its samples/truth are in `results/`.

## Conventions that matter here

- Compare predictors on identical runs (paired). Run-level SEs over `sample_run` draws understate uncertainty because runs reuse the same 221 pairs — report pair-cluster bootstrap SEs too, and remember there are only five public benchmarks. Formative leaderboard scores are single noisy draws (single-run ALC sd ~0.02–0.04).
- Priors and hyperparameters must be fitted leave-one-benchmark-out relative to the run being scored; the hidden test is a benchmark-level cold start.
- Negative results are recorded as negative in `docs/`, each with the script that produced it. Keep docs numbers in sync with a runnable script.
- Recurring bug signature: a spike in Brier at budgets 1 and 3 means the second-order term is mishandled (MAP without Laplace integration, standardizing after shrinkage, a Hessian missing a factor `n`). Check that first when low-budget numbers look wrong.
- Only binary item-level benchmarks are eligible in the test; `mmdocrag` (fraction) is excluded, `matharena` ('mixed') is used after dropping its non-binary responses, `swe_rebench` has a single subject.
