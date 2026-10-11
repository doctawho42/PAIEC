"""Wall time of paiec.prior.benchmark_fit on pool benchmarks of several sizes.

Usage: python time_fits.py B[:max_pairs[:max_items]] ...
Prints one JSON line per benchmark: pairs, items, cells, columns, EM
iterations, convergence and seconds (load and fit separately). Fits nothing
but the per-benchmark variance-components model; scores nothing.
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto_loader import load_benchmark  # noqa: E402
from paiec import prior as PR  # noqa: E402


def main():
    for arg in sys.argv[1:]:
        parts = arg.split(":")
        b = parts[0]
        mp = int(parts[1]) if len(parts) > 1 and parts[1] else None
        mi = int(parts[2]) if len(parts) > 2 and parts[2] else None
        t0 = time.perf_counter()
        pairs = load_benchmark(b, mp, mi)
        t1, c1 = time.perf_counter(), time.process_time()
        if not pairs:
            print(json.dumps({"benchmark": b, "pairs": 0}), flush=True)
            continue
        if len(pairs) >= 2:
            f = PR.benchmark_fit(pairs)
        else:
            f = PR.benchmark_fit(pairs, s2e=7.0)
        t2, c2 = time.perf_counter(), time.process_time()
        print(json.dumps({"benchmark": b, "max_pairs": mp, "max_items": mi, "pairs": f["n_pairs"],
                          "items": f["n_items"], "cells": f["n_cells"], "keys": f["keys"],
                          "iterations": f["iterations"], "converged": f["converged"],
                          "newton_converged": f["newton_converged"],
                          "mu": round(f["mu"], 3), "s2_t": f["s2_t"] and round(f["s2_t"], 3),
                          "s2e": round(f["s2e"], 3),
                          "load_s": round(t1 - t0, 2), "fit_s": round(t2 - t1, 2),
                          "fit_cpu_s": round(c2 - c1, 2), "load_avg": os.getloadavg()[0]}), flush=True)


if __name__ == "__main__":
    main()
