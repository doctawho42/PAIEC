"""Leave-one-benchmark-out shrinkage slope of a name's residual cells when the
other cells come only from benchmarks with a different modal harness string
(a different evaluation pipeline), and which benchmark pairs carry the names
seen on exactly two benchmarks. Descriptive; scores nothing. Output:
code_harness_slopes.txt (run 2026-10-07 inline with the same code)."""
import sys, pickle, collections, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_variants import rows_of, residual_cells, PD
fits = pickle.load(open(os.path.join(PD, "fits_K100_M2000.pkl"), "rb"))
rows = rows_of(fits)
cells = residual_cells(rows)
by_k = collections.defaultdict(dict)
for (k, b), v in cells.items():
    by_k[k][b] = v
pairs = collections.Counter(tuple(sorted(d)) for d in by_k.values() if len(d) == 2)
print(pairs.most_common(8))
harn = {}
for b, f in fits.items():
    harn[b] = collections.Counter(s.get("harness", "") for s in f["subjects"]).most_common(1)[0][0]
xy = []
for k, d in by_k.items():
    for b, y in d.items():
        others = [v for bb, v in d.items() if bb != b and (harn[bb] != harn[b] or not harn[b])]
        if others:
            xy.append((np.mean(others), y, len(others)))
a = np.array(xy)
for name, sel in (("all", np.ones(len(a), bool)), ("k1", a[:, 2] == 1),
                  ("k2-4", (a[:, 2] >= 2) & (a[:, 2] <= 4)), ("k5+", a[:, 2] >= 5)):
    x, y = a[sel, 0] - a[sel, 0].mean(), a[sel, 1] - a[sel, 1].mean()
    print("other cells with a different harness string:", name, int(sel.sum()), round(float(x @ y / (x @ x)), 3))
print(collections.Counter(harn.values()).most_common(6))
