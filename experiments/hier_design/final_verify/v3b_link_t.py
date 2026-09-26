import sys
from v3_link import exact_link, labeled, CASES
from common import HierPredictor, Hyper
from paiec import prior as PR
REFT = Hyper(**{**Hyper().to_dict(), **PR.REFERENCE, "nu_mu": 3.0})
for label, h in (("t3 default", Hyper(sigma_mu=1.829, nu_mu=3.0)), ("t3 REFERENCE", REFT),
                 ("gauss REFERENCE", Hyper(**{**Hyper().to_dict(), **PR.REFERENCE}))):
    print(label, "link weight w/o attributes", (h.sigma_theta**2 + h.sigma_attr**2) / (h.sigma_theta**2 + h.sigma_attr**2 + h.sigma_delta**2))
    for T1, T2, O1, O2 in CASES:
        ex = exact_link(h, T1, T2, O1, O2)
        lab, inp = labeled(T1, T2, O1, O2)
        m = HierPredictor(None, h, floor=False, slip=False)
        p = m.predict(inp, lab)
        print(f"{label:<16} T1 {T1} T2 {T2} O1 {O1} O2 {O2}: exact {ex:.4f} line {p:.4f} ({p-ex:+.4f}) unconv {m.unconverged}", flush=True)
