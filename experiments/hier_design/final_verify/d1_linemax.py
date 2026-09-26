"""Dense runs simulated from the model: predictions with the line capped at
LINE_MAX (default 600) against the line uncapped, per budget."""
import sys, time
import numpy as np
from common import HierPredictor, Hyper
import paiec.hier as H
from sim import simulate

h = Hyper()
n_s = int(sys.argv[1]) if len(sys.argv) > 1 else 40
scope = sys.argv[2] if len(sys.argv) > 2 else "pair"
rng = np.random.default_rng(11)
for rep in range(3):
    lab, targets = simulate(rng, h, [(s, 0) for s in range(n_s)], n_items=160, scope=scope)
    ys = np.array([y for _, y, _ in targets])
    for B in (1, 3, 7, 15, 31):
        res = {}
        for cap in (600, 10**9):
            H.LINE_MAX = cap
            m = HierPredictor(None, h, floor=False, slip=False)
            t = time.perf_counter()
            p = np.array([m.predict(inp, lab[B]) for inp, _, _ in targets])
            res[cap] = (p, time.perf_counter() - t)
        d = res[10**9][0] - res[600][0]
        # per pair mean diff
        pid = np.array([pi for _, _, pi in targets])
        pd = np.array([d[pid == k].mean() for k in range(n_s)])
        print(f"rep{rep} B{B:<2} labels {len(lab[B]):4d}: |diff| mean {np.abs(d).mean():.4f} max {np.abs(d).max():.4f}; "
              f"per-pair max |mean diff| {np.abs(pd).max():.4f}; Brier cap {np.mean((res[600][0]-ys)**2):.5f} "
              f"uncapped {np.mean((res[10**9][0]-ys)**2):.5f}; time {res[600][1]:.1f}s vs {res[10**9][1]:.1f}s", flush=True)
