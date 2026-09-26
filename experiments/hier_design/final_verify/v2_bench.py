"""Two pairs, a new subject after others, labeled-item targets: hier (line on/off,
Gaussian / t level, floor on/off) against exact_bench."""
import sys, time
import numpy as np
from common import HierPredictor, Hyper, item, mcq, subject
from exact1 import exact_bench

BID = "benchmark_1"


def build(others, target, star_target, floor):
    mk = (lambda j: mcq(j, BID)) if floor else (lambda j: item(j, BID))
    lab, j = [], 0
    for o, (k, n, star) in enumerate(others):
        s = subject(f"other {o}")
        for q in range(n):
            lab.append([[s, mk(j)], int(q < k)]); j += 1
        for y in star:
            lab.append([[s, mk(9999)], int(y)])
    T = subject("target")
    kT, nT = target
    for q in range(nT):
        lab.append([[T, mk(j)], int(q < kT)]); j += 1
    return lab, [T, mk(9999 if star_target else 8888)]


CASES = [
    # name, others, target (k, n), star_target
    ("2 pairs: other 0/7, T 1/1", [(0, 7, [])], (1, 1), False),
    ("2 pairs: other 7/7, T 0/3", [(7, 7, [])], (0, 3), False),
    ("2 pairs: other 3/15, T 15/15", [(3, 15, [])], (15, 15), False),
    ("2 pairs: other 31/31, T 2/31", [(31, 31, [])], (2, 31), False),
    ("new subj after 6x 0/31", [(0, 31, [])] * 6, (0, 0), False),
    ("new subj after 6x 31/31", [(31, 31, [])] * 6, (0, 0), False),
    ("new subj after 3x 1/7", [(1, 7, [])] * 3, (0, 0), False),
    ("T 1/1 after 6x 0/31", [(0, 31, [])] * 6, (1, 1), False),
    ("T 0/3 after 6x 2/31", [(2, 31, [])] * 6, (0, 3), False),
    ("T 3/7 after 12x 5/15", [(5, 15, [])] * 12, (3, 7), False),
    ("item*: 5 others fail it (0/7 each), T new", [(0, 7, [0])] * 5, (0, 0), True),
    ("item*: 5 others fail it (3/7 each), T 3/7", [(3, 7, [0])] * 5, (3, 7), True),
    ("item*: 3 pass 1 fail, T 0/3", [(2, 7, [1])] * 3 + [(2, 7, [0])], (0, 3), True),
    ("item*: 1 other passes, T 1/1", [(0, 3, [1])], (1, 1), True),
    ("item*: 6 others fail, others 31/31, T 31/31", [(31, 31, [0])] * 6, (31, 31), True),
    ("item*: 6 pass, others 0/31, T 0/31", [(0, 31, [1])] * 6, (0, 31), True),
]
configs = {
    "gauss": (Hyper(), False),
    "gauss+floor": (Hyper(), True),
    "t3 1.829": (Hyper(sigma_mu=1.829, nu_mu=3.0), False),
    "t3+floor": (Hyper(sigma_mu=1.829, nu_mu=3.0), True),
}
which = sys.argv[1:] or list(configs)
for cname in (which if __name__ == "__main__" else []):
    h, floor = configs[cname]
    c = h.guess * 0.25 if floor else 0.0
    for name, others, target, star in CASES:
        t = time.perf_counter()
        ex = exact_bench(h, others, target, star, c=c)
        ex = c + (1 - c) * ex
        te = time.perf_counter() - t
        lab, inp = build(others, target, star, floor)
        res = []
        for line in (True, False):
            m = HierPredictor(None, h, slip=False, line=line)
            p = m.predict(inp, lab)
            res.append((p, m.failures, m.unconverged))
        print(f"{cname:<12} {name:<46} exact {ex:.4f}  line {res[0][0]:.4f} ({res[0][0]-ex:+.4f})  "
              f"laplace {res[1][0]:.4f} ({res[1][0]-ex:+.4f})  fail {res[0][1]}/{res[1][1]} unconv {res[0][2]}/{res[1][2]}  [{te:.0f}s]", flush=True)
