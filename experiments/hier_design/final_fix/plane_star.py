import sys, time
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # the repo
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "final_verify"))
sys.path.insert(0, _HERE)
import plane
from common import HierPredictor, Hyper
from exact1 import exact_bench
from v2_bench import build, CASES
configs = {"gauss": (Hyper(), False), "gauss+floor": (Hyper(), True)}
for cname in sys.argv[1:] or ["gauss"]:
    h, floor = configs[cname]
    c = h.guess * 0.25 if floor else 0.0
    for name, others, target, star in CASES:
        if not star:
            continue
        ex = c + (1 - c) * exact_bench(h, others, target, star, c=c)
        lab, inp = build(others, target, star, floor)
        plane.uninstall()
        pl = HierPredictor(None, h, slip=False, floor=floor).predict(inp, lab)
        plane.install()
        t = time.perf_counter()
        m = HierPredictor(None, h, slip=False, floor=floor)
        pp = m.predict(inp, lab)
        dt = time.perf_counter() - t
        plane.uninstall()
        print(f"{cname:<12} {name:<46} exact {ex:.4f} line {pl-ex:+.4f} plane {pp-ex:+.4f} fail {m.failures} ({dt:.2f}s) {plane.STATS}", flush=True)
