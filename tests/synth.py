"""Synthetic pairs that mimic the real thing closely enough to test the protocol."""
import numpy as np
from paiec.evaluator import Pair, Response

sig = lambda x: 1 / (1 + np.exp(-x))


def make_pair(subject_id="s0", benchmark_id="b0", n_items=100, seed=0,
              repeats=1, mcq_frac=0.35, theta=None):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0.2, 0.45, n_items))
    b = rng.normal(0.0, 1.3, n_items)
    c = np.where(rng.random(n_items) < mcq_frac, 0.25, 0.0)
    d = 0.98
    th = rng.normal(0, 1) if theta is None else theta
    p = c + (d - c) * sig(a * (th - b))
    subject = {"normalized_name": subject_id, "provider": "acme",
               "release_date": "2025-01-01", "access_date": "2026-01-01",
               "harness": "default", "harness_version": "", "reasoning_effort": "",
               "subject_features_extra": ""}
    responses = []
    for i in range(n_items):
        item = {"item_content": f"item {i} of {benchmark_id}",
                "item_features": f"tier=x;idx={i}", "interactors": "",
                "benchmark_id": benchmark_id}
        for _ in range(repeats):
            responses.append(Response(item_key=f"{benchmark_id}:{i}", item=item,
                                      label=int(rng.random() < p[i])))
    pr = Pair(subject=subject, subject_id=subject_id, benchmark_id=benchmark_id,
              responses=responses)
    pr.truth = {f"{benchmark_id}:{i}": float(p[i]) for i in range(n_items)}
    pr.theta = float(th)
    return pr


def make_pairs(n_subjects=6, n_benchmarks=3, n_items=100, seed=0, repeats=1):
    out = []
    for s in range(n_subjects):
        th = np.random.default_rng(1000 + s).normal(0, 1)
        for b in range(n_benchmarks):
            out.append(make_pair(f"s{s}", f"b{b}", n_items,
                                 seed=seed + 97 * s + 13 * b, repeats=repeats, theta=th))
    return out
