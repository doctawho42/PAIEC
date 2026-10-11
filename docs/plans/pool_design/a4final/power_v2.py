"""Planning power of gate condition (a) under the revised split (amendment 4
final, recommended change 1: Kish n in place of exp(entropy)). Arithmetic on
the tilt weights of split_v2.json's chosen knobs; scores nothing.

SE = sigma_b / sqrt(n), n the Kish number of the regime's weights aggregated
to source families (the bootstrap's cluster); one regime passes (a) with
probability Phi((Delta - max(0.002, 1.96 SE)) / SE).
Run:  PYTHONDONTWRITEBYTECODE=1 python power_v2.py   -> power_v2.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import split_v2 as SV  # noqa: E402


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    d = json.load(open(os.path.join(HERE, "split_v2.json")))
    half = SV.split("family")
    out = {}
    for H in ("S", "C"):
        bs = [b for b in SV.E if half[b] == H]
        for r in ("P-READING", "P-AUDIT", "P-TUNED", "P-MIXTURE", "D-READING"):
            comps = d["results"]["family"]["cap"][H][r]["knobs"]
            w = SV.weights(bs, [tuple(c) for c in comps], SV.CAP_SHARE)
            fam = {}
            for b, x in zip(bs, w):
                fam[SV.FAM_OF[b]] = fam.get(SV.FAM_OF[b], 0) + x
            out.setdefault(H, {})[r] = {"kish_benchmarks": round(float(1 / np.sum(w ** 2)), 1),
                                        "kish_families": round(float(1 / sum(v * v for v in fam.values())), 1)}
    tab = []
    for sb in (0.005, 0.010, 0.020):
        row = {"sigma_b": sb}
        for r in ("P-READING", "P-AUDIT"):
            n = out["C"][r]["kish_families"]
            se = sb / math.sqrt(n)
            row[r] = {"SE": round(se, 5), **{f"pass_at_{D}": round(Phi((D - max(0.002, 1.96 * se)) / se), 3)
                                              for D in (0.002, 0.004, 0.006)},
                      "false_pass_at_0": round(Phi((0 - max(0.002, 1.96 * se)) / se), 4)}
        tab.append(row)
    out["power_C"] = tab
    with open(os.path.join(HERE, "power_v2.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
