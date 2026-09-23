"""How much of the pooling gain survives when a benchmark has few subjects?

Design: per benchmark and replicate, permute the subjects once and take nested
prefixes S_3 subset S_5 subset ... subset S_all. Score ONLY the pairs in the
smallest prefix, every time. The scored pairs are then identical across pool
sizes within a replicate, so the only thing that varies is how many other
subjects contribute labels to the pool.
"""
import numpy as np
from collections import defaultdict
from paiec import evaluator as E
from paiec.irt import fit_joint, text_embeddings
from experiments._pooled_legacy import fit_ab

sig = lambda x: 1 / (1 + np.exp(-x))
N_SCORED = 3


def _score(pool_pairs, scored_pairs, traj, X, kidx, B, lam_t, lam_w, lam_e, seed):
    sidx = {(p.subject_id, p.benchmark_id): i for i, p in enumerate(pool_pairs)}
    si, ji, y = [], [], []
    for p in pool_pairs:
        for k, lab in traj[(p.subject_id, p.benchmark_id)][:B]:
            si.append(sidx[(p.subject_id, p.benchmark_id)]); ji.append(kidx[k]); y.append(lab)
    n_s, n_i, d = len(pool_pairs), X.shape[0], X.shape[1]
    if len(y) and len(set(y)) > 1:
        si, ji, y = np.array(si), np.array(ji), np.array(y, float)
        th, w, eps = fit_joint(si, ji, y, X, n_s, n_i, lam_t, lam_w, lam_e)
        b = X @ w + eps
        s = np.clip(sig(th[si] - b[ji]), 1e-6, 1 - 1e-6); s = s * (1 - s)
        Xo = X[ji]
        Sw = np.linalg.inv(lam_w * np.eye(d) + Xo.T @ (Xo * s[:, None]) + 1e-9 * np.eye(d))
        veps = 1.0 / (lam_e + np.bincount(ji, weights=s, minlength=n_i))
        vb = np.einsum("ij,jk,ik->i", X, Sw, X) + veps
        zb = -(b - b.mean()) / np.sqrt(1.0 + (np.pi / 8.0) * vb)
    else:
        zb = np.zeros(n_i)
    out = []
    for p in scored_pairs:
        rows = traj[(p.subject_id, p.benchmark_id)][:B]
        a, bb = fit_ab(np.array([zb[kidx[k]] for k, _ in rows]),
                       np.array([v for _, v in rows], float)) if rows else (0.0, 1.0)
        _, ev = E.split_pair(p, seed); evs = set(ev)
        se = n = 0.0
        for r in p.responses:
            if r.item_key not in evs:
                continue
            se += (np.clip(sig(a + bb * zb[kidx[r.item_key]]), 1e-4, 1 - 1e-4) - r.label) ** 2
            n += 1
        out.append(se / n)
    return out


def smoothed_ref(scored_pairs, traj, B, prior_n=4.0, seed=0):
    out = []
    for p in scored_pairs:
        rows = traj[(p.subject_id, p.benchmark_id)][:B]
        s = sum(v for _, v in rows)
        pr = (s + prior_n * 0.5) / (len(rows) + prior_n)
        _, ev = E.split_pair(p, seed); evs = set(ev)
        se = n = 0.0
        for r in p.responses:
            if r.item_key not in evs:
                continue
            se += (pr - r.label) ** 2; n += 1
        out.append(se / n)
    return out


def run(pairs, traj, sizes=(3, 5, 10, 20, 40, 80), reps=8, dim=64,
        lam_t=1.0, lam_w=64.0, lam_e=1.0, seed=0):
    by_b = defaultdict(list)
    for p in pairs:
        by_b[p.benchmark_id].append(p)
    emb, kidxs = {}, {}
    for bid, ps in by_b.items():
        seen, order, txt = set(), [], []
        for p in ps:
            for r in p.responses:
                if r.item_key not in seen:
                    seen.add(r.item_key); order.append(r.item_key); txt.append(r.item["item_content"] or "")
        kidxs[bid] = {k: i for i, k in enumerate(order)}
        emb[bid] = text_embeddings(txt, dim=dim)
    acc = defaultdict(list); ref = defaultdict(list)
    for bid, ps in by_b.items():
        if len(ps) < max(N_SCORED + 1, 5):
            continue
        for rep in range(reps):
            rng = np.random.default_rng(1000 * rep + hash(bid) % 1000)
            perm = [ps[i] for i in rng.permutation(len(ps))]
            scored = perm[:N_SCORED]
            for N in sizes:
                if N > len(perm):
                    continue
                pool = perm[:N]
                for B in E.BUDGETS:
                    acc[(N, B)] += _score(pool, scored, traj, emb[bid], kidxs[bid],
                                          B, lam_t, lam_w, lam_e, seed)
            for B in E.BUDGETS:
                ref[B] += smoothed_ref(scored, traj, B)
    return acc, ref
