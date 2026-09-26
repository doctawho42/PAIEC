"""Part 1: benchmark levels on the pair-accuracy scale, and what prior they imply.

  1. Per benchmark: mean and sd of pair logit accuracy, and a logit-normal-
     binomial ML fit (level m_b, within-benchmark sd s_b) that does not let the
     near-0 pairs blow up the logits.
  2. Between benchmarks: mu0 and sigma_mu from the five levels, leave-one-out
     levels, and the predictive for a new benchmark.
  3. How much of the within-benchmark sd the attribute prior removes, leave one
     benchmark out (the residual is the sd the level prior still has to carry).
  4. E[p(1-p)] of a logit-normal pair distribution: what spread or centre gives
     the 0.128 the leaderboard hint implies.

Writes levels.json. Run: python levels.py
"""
import json

from scipy import optimize, stats

from common import *  # noqa: F401,F403
from score import load_runs

GX, GW = np.polynomial.hermite_e.hermegauss(40)
GW = GW / GW.sum()


def epq(m, s):
    """E[p(1-p)] for logit p ~ N(m, s^2)."""
    p = sig(m + s * GX)
    return float(np.sum(GW * p * (1 - p)))


def emean(m, s):
    return float(np.sum(GW * sig(m + s * GX)))


def lnb_fit(k, n):
    """ML of (m, s) for k_j ~ Binomial(n_j, sigmoid(m + s z)), z ~ N(0,1);
    returns estimates and SEs from the observed information."""
    k, n = np.asarray(k, float), np.asarray(n, float)

    def nll(t):
        m, ls = t
        P = np.clip(sig(m + np.exp(ls) * GX[None, :]), 1e-12, 1 - 1e-12)
        L = k[:, None] * np.log(P) + (n - k)[:, None] * np.log(1 - P)
        mx = L.max(1, keepdims=True)
        return -float(np.sum(np.log(np.sum(GW * np.exp(L - mx), 1)) + mx[:, 0]))

    r = optimize.minimize(nll, [logit(np.clip(k.sum() / n.sum(), .01, .99)), 0.0], method="Nelder-Mead",
                          options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
    h, t = 1e-3, r.x
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i, e_j = np.eye(2)[i] * h, np.eye(2)[j] * h
            H[i, j] = (nll(t + e_i + e_j) - nll(t + e_i - e_j) - nll(t - e_i + e_j) + nll(t - e_i - e_j)) / (4 * h * h)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return float(t[0]), float(np.exp(t[1])), float(se[0]), float(se[1])


def pair_table(df):
    g = df.groupby(["bench", "subject"])
    t = pd.DataFrame({"p": g.y.mean(), "n_resp": g.y.size(), "n_items": g.item.nunique()}).reset_index()
    t["k_items"] = t.p * t.n_items          # effective successes on the item scale
    t["logit"] = logit(np.clip((t.p * t.n_items + 0.5) / (t.n_items + 1), 1e-6, 1 - 1e-6))
    return t


def attribute_residual(pairs, t):
    """LOBO: fit the shipped attribute prior without benchmark b, predict b's
    pairs, compare with their centred logits."""
    from paiec.predict import Predictor, fit_prior
    from paiec.subjects import subject_frame
    sf = subject_frame(pairs)
    out = {}
    for b in sorted(sf.benchmark_id.unique()):
        held = sf[sf.benchmark_id == b]
        if len(held) < 5:
            continue
        coef, spec = fit_prior([p for p in pairs if p.benchmark_id != b])
        pr = Predictor(coef, spec)
        z_hat = np.array([pr.prior_mean(p.subject) for p in pairs if p.benchmark_id == b])
        z = held.z.values
        slope = float(np.polyfit(z_hat, z, 1)[0])
        out[b] = {"n": int(len(z)), "sd_z": float(z.std()), "corr": float(np.corrcoef(z, z_hat)[0, 1]),
                  "resid_sd_slope1": float(np.std(z - z_hat)), "resid_sd_best_slope": float(np.std(z - slope * z_hat)),
                  "best_slope": slope, "sd_zhat": float(z_hat.std())}
    return out


def main():
    df = flat()
    pairs = eligible_pairs()
    t = pair_table(df)
    out = {"benchmarks": {}}
    for b, g in t.groupby("bench"):
        row = {"pairs": int(len(g)), "mean_p": float(g.p.mean()), "mean_logit": float(g.logit.mean()),
               "sd_logit": float(g.logit.std()) if len(g) > 1 else None,
               "logit_of_mean_p": float(logit(g.p.mean()))}
        if len(g) > 2:
            m, s, sem, ses = lnb_fit(g.k_items, g.n_items)
            row.update({"lnb_m": m, "lnb_m_se": sem, "lnb_s": s, "lnb_s_se": ses})
        else:
            row.update({"lnb_m": row["mean_logit"], "lnb_m_se": 0.1, "lnb_s": None})
        out["benchmarks"][b] = row
        print(b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    lv = np.array([r["lnb_m"] for r in out["benchmarks"].values()])
    se2 = np.array([r["lnb_m_se"] ** 2 for r in out["benchmarks"].values()])
    K = len(lv)
    mu0, var_obs = float(lv.mean()), float(lv.var(ddof=1))
    s_mu = float(np.sqrt(max(var_obs - se2.mean(), 0)))
    lo, hi = np.sqrt((K - 1) * var_obs / stats.chi2.ppf([0.975, 0.025], K - 1))
    within = [r["lnb_s"] for r in out["benchmarks"].values() if r["lnb_s"]]
    w_n = [r["pairs"] for r in out["benchmarks"].values() if r["lnb_s"]]
    s_u = float(np.sqrt(np.average(np.square(within), weights=w_n)))
    names = list(out["benchmarks"])
    out["between"] = {
        "levels": dict(zip(names, lv.tolist())), "mu0": mu0, "mu0_se": float(np.sqrt(var_obs / K)),
        "sd_levels": float(np.sqrt(var_obs)), "sigma_mu_net_of_se": s_mu, "sigma_mu_95": [float(lo), float(hi)],
        "loo_mu0": {n: float(np.delete(lv, j).mean()) for j, n in enumerate(names)},
        "loo_abs_error": float(np.mean([abs(lv[j] - np.delete(lv, j).mean()) for j in range(K)])),
        "loo_rmse": float(np.sqrt(np.mean([(lv[j] - np.delete(lv, j).mean()) ** 2 for j in range(K)]))),
        "predictive_sd_new_benchmark_t4": float(np.sqrt(var_obs * (1 + 1 / K))),
        "sigma_u_pooled": s_u,
        "total_sd_pair_logit": float(np.sqrt(s_mu ** 2 + s_u ** 2)),
        "p_at_mu0": float(sig(mu0)), "mean_p_implied": emean(mu0, np.sqrt(s_mu ** 2 + s_u ** 2)),
        "epq_implied": epq(mu0, np.sqrt(s_mu ** 2 + s_u ** 2)),
    }
    print("between", out["between"])

    out["attribute_prior_lobo"] = attribute_residual(pairs, t)
    for b, r in out["attribute_prior_lobo"].items():
        print("attributes", b, {k: round(v, 3) for k, v in r.items()})
    ar = out["attribute_prior_lobo"]
    tot_n = sum(r["n"] for r in ar.values())
    out["attribute_resid_sd_pooled"] = float(np.sqrt(sum(r["n"] * r["resid_sd_slope1"] ** 2 for r in ar.values()) / tot_n))
    out["attribute_raw_sd_pooled"] = float(np.sqrt(sum(r["n"] * r["sd_z"] ** 2 for r in ar.values()) / tot_n))

    # R1 pair appearances: the distribution the formative-like runs score
    runs = load_runs()
    pe = np.array([p["p_eval"] for r in runs for p in r])
    wr = np.concatenate([np.full(len(r), 1 / len(r)) for r in runs])
    lg = logit(np.clip(pe, 0.005, 0.995))
    m_r1 = float(np.average(lg, weights=wr))
    s_r1 = float(np.sqrt(np.average((lg - m_r1) ** 2, weights=wr)))
    q_r1 = float(np.average(pe * (1 - pe), weights=wr))
    # the logit-normal matching R1's E[p] and E[p(1-p)] (moment match)
    mean_r1 = float(np.average(pe, weights=wr))
    f = lambda x: [emean(x[0], abs(x[1])) - mean_r1, epq(x[0], abs(x[1])) - q_r1]
    mm = optimize.fsolve(f, [m_r1, s_r1])
    m_mm, s_mm = float(mm[0]), float(abs(mm[1]))
    target = 0.128
    s_star = optimize.brentq(lambda s: epq(m_mm, s) - target, 0.01, 10)
    m_low = optimize.brentq(lambda m: epq(m, s_mm) - target, -8, m_mm)
    m_high = optimize.brentq(lambda m: epq(m, s_mm) - target, m_mm, 8)
    out["r1"] = {"mean_p": mean_r1, "E_pq": q_r1, "mean_logit": m_r1, "sd_logit": s_r1,
                 "moment_matched_logit_normal": [m_mm, s_mm],
                 "for_E_pq_0.128": {"sd_at_same_centre": float(s_star), "sd_ratio": float(s_star / s_mm),
                                    "centre_low": float(m_low), "centre_high": float(m_high),
                                    "mean_p_low": emean(m_low, s_mm), "mean_p_high": emean(m_high, s_mm)}}
    print("r1", out["r1"])
    grid = {f"{m:+.1f},{s:.1f}": epq(m, s) for m in (-2, -1.5, -1, -0.6, 0, 0.6) for s in (0.5, 1, 1.5, 2, 2.5, 3)}
    out["epq_grid"] = grid
    with open(os.path.join(HERE, "levels.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
