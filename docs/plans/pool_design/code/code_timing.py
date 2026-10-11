"""How the cost of paiec.prior.benchmark_fit grows with the pair and item caps.

Not repository code. For code.md only: fits the per-benchmark variance-components
model for a few EM iterations (max_iter) at several caps and reports CPU seconds
per EM iteration, peak RSS, columns and cells, so that the full-fit cost can be
extrapolated with the iteration counts of the K100/M2000 prototype. Scores
nothing, compares no predictors, reads no item text.

Usage: python code_timing.py B:K:M:ITER ...   (K or M empty = no cap)
"""
import json
import os
import resource
import sys
import time

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from proto_loader import load_benchmark  # noqa: E402
from paiec import prior as PR  # noqa: E402


def main():
    for arg in sys.argv[1:]:
        b, k, m, it = (arg.split(":") + ["", "", "3"])[:4]
        K, M, n_it = (int(k) if k else None), (int(m) if m else None), int(it)
        c0, t0 = time.process_time(), time.perf_counter()
        pairs = load_benchmark(b, K, M)
        c1, t1 = time.process_time(), time.perf_counter()
        f = PR.benchmark_fit(pairs, max_iter=n_it) if len(pairs) >= 2 else \
            PR.benchmark_fit(pairs, s2e=5.8, max_iter=n_it)
        c2, t2 = time.process_time(), time.perf_counter()
        print(json.dumps({"benchmark": b, "K": K, "M": M, "pairs": f["n_pairs"], "items": f["n_items"],
                          "cells": f["n_cells"], "columns": 1 + f["n_pairs"] + 0, "keys": f["keys"],
                          "em_iterations": f["iterations"], "load_cpu_s": round(c1 - c0, 2),
                          "fit_cpu_s": round(c2 - c1, 2), "fit_wall_s": round(t2 - t1, 2),
                          "cpu_per_em_iteration_s": round((c2 - c1) / max(f["iterations"], 1), 3),
                          "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20),
                          "load_avg": round(os.getloadavg()[0], 1)}), flush=True)
        PR._FITS.clear()


if __name__ == "__main__":
    main()
