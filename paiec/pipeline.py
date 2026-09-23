"""Two-pass evaluation of the acquisition policy.

Pass 1 builds trajectories with the evaluator's random policy and fits the
pooled IRT on them, which gives a difficulty scale to steer by. Pass 2 re-walks
the same candidate order with the informed policy. The final score refits the
pooled IRT on the pass-2 trajectories, so the policy is judged on the labels it
actually chose.

The approximation: pass 2 steers by a scale built from full-budget pass-1
labels. A real submission processes pairs in sequence, so later pairs do see
earlier pairs' full trajectories; early ones see less.
"""
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
from paiec import evaluator as E
from paiec import baselines as Bs
from paiec.data import load_pairs
from paiec.trajectories import collect_trajectories
from paiec.subjects import subject_frame, design
from paiec.irt import fit_joint, text_embeddings
from paiec.fitting import fit_ab, sig
from paiec.mcq import floor_of
from paiec import acquisition as acq


def build(pairs):
    df = subject_frame(pairs)
    prov = [p for p, c in df.provider.value_counts().items() if c >= 8 and p != "unknown"]
    eff = [e for e, c in df.effort.value_counts().items() if c >= 8 and e != "none"]
    zh = {}
    for b in df.benchmark_id.unique():
        tr, te = df[df.benchmark_id != b], df[df.benchmark_id == b]
        if len(tr) < 20:
            for r in te.itertuples():
                zh[(r.subject_id, r.benchmark_id)] = 0.0
            continue
        m = Ridge(alpha=2.0).fit(design(tr, prov, eff), tr.z.values)
        for r, z in zip(te.itertuples(), m.predict(design(te, prov, eff))):
            zh[(r.subject_id, r.benchmark_id)] = float(z)
    by_b = defaultdict(list)
    for p in pairs:
        by_b[p.benchmark_id].append(p)
    keys_of, emb = {}, {}
    for bid, ps in by_b.items():
        seen, order, txt = set(), [], []
        for p in ps:
            for r in p.responses:
                if r.item_key not in seen:
                    seen.add(r.item_key); order.append(r.item_key)
                    txt.append(r.item["item_content"] or "")
        keys_of[bid] = {k: i for i, k in enumerate(order)}
        emb[bid] = text_embeddings(txt, dim=64)
    return zh, by_b, keys_of, emb


def pooled_z(ps, traj, X, kidx, B, lam_t=1.0, lam_w=64.0, lam_e=1.0):
    sidx = {(p.subject_id, p.benchmark_id): i for i, p in enumerate(ps)}
    si, ji, y = [], [], []
    for p in ps:
        for k, l in traj[(p.subject_id, p.benchmark_id)][:B]:
            si.append(sidx[(p.subject_id, p.benchmark_id)]); ji.append(kidx[k]); y.append(l)
    if not y or len(set(y)) < 2:
        return np.zeros(len(X))
    si, ji, y = np.array(si), np.array(ji), np.array(y, float)
    th, w, eps = fit_joint(si, ji, y, X, len(ps), len(X), lam_t, lam_w, lam_e)
    b = X @ w + eps
    s = np.clip(sig(th[si] - b[ji]), 1e-6, 1 - 1e-6); s = s * (1 - s)
    Xo = X[ji]; d = X.shape[1]
    Sw = np.linalg.inv(lam_w * np.eye(d) + Xo.T @ (Xo * s[:, None]) + 1e-9 * np.eye(d))
    veps = 1.0 / (lam_e + np.bincount(ji, weights=s, minlength=len(X)))
    vb = np.einsum("ij,jk,ik->i", X, Sw, X) + veps
    return -(b - b.mean()) / np.sqrt(1 + (np.pi / 8) * vb)


def score(pairs, traj, zh, by_b, keys_of, emb, slip=0.02, temp=None, seed=0,
          v_a=2.0, v_b=0.25):
    out = {}
    for B in E.BUDGETS:
        se_all = []
        for bid, ps in by_b.items():
            kidx, X = keys_of[bid], emb[bid]
            zb = pooled_z(ps, traj, X, kidx, B)
            for p in ps:
                rows = traj[(p.subject_id, p.benchmark_id)][:B]
                Z = np.array([zb[kidx[k]] for k, _ in rows])
                Y = np.array([v for _, v in rows], float)
                m_a = zh.get((p.subject_id, p.benchmark_id), 0.0)
                a, b, va, vb = fit_ab(Z, Y, m_a, v_a, 1.0, v_b)
                _, ev = E.split_pair(p, seed); evs = set(ev); se = n = 0.0
                for r in p.responses:
                    if r.item_key not in evs:
                        continue
                    z = zb[kidx[r.item_key]]
                    k = 1 / np.sqrt(1 + (np.pi / 8) * (va + vb * z * z))
                    eta = k * (a + b * z)
                    if temp is not None:
                        eta *= temp[B]
                    c = floor_of(r.item["item_content"] or "")
                    se += (np.clip(c + (1 - c - slip) * sig(eta), 1e-4, 1 - 1e-4) - r.label) ** 2
                    n += 1
                se_all.append(se / n)
        out[B] = float(np.mean(se_all))
    return out


def informed_traj(pairs, base_traj, zh, by_b, keys_of, emb, gamma=0.5, seed=0):
    new = {}
    for bid, ps in by_b.items():
        kidx, X = keys_of[bid], emb[bid]
        zb = pooled_z(ps, base_traj, X, kidx, 31)
        for p in ps:
            acq_keys, _ = E.split_pair(p, seed)
            rng = np.random.default_rng(
                (seed * 7 + E.stable_hash(p.subject_id, p.benchmark_id, "order")) % (2**32))
            order = [acq_keys[i] for i in rng.permutation(len(acq_keys))]
            by = p.by_item()
            oz = np.array([zb[kidx[k]] for k in order])
            oy = [by[k][0].label for k in order]
            m_a = zh.get((p.subject_id, p.benchmark_id), 0.0)
            idx = acq.stream(oz, oy, m_a, gamma=gamma)
            new[(p.subject_id, p.benchmark_id)] = [(order[i], oy[i]) for i in idx]
    return new


def coverage_traj(pairs, base_traj, zh, by_b, keys_of, emb, seed=0, w_info=0.0,
                  max_labels=31, warmup=10):
    """Pairs are processed in sequence and each prefers items the pool has not
    touched yet. Random acquisition wastes labels on collisions: 81 subjects x 31
    labels over 1633 items reach only 899 distinct items. Coordinating on
    coverage is what the shared difficulty estimate actually wants.

    Needs the running pool to be visible during acquisition, so it stands or
    falls with the same open question as everything else here.
    """
    new = {}
    for bid, ps in by_b.items():
        kidx, X = keys_of[bid], emb[bid]
        zb = pooled_z(ps, base_traj, X, kidx, 31)
        count = defaultdict(int)
        for p in ps:
            acq_keys, _ = E.split_pair(p, seed)
            rng = np.random.default_rng(
                (seed * 7 + E.stable_hash(p.subject_id, p.benchmark_id, "order")) % (2**32))
            order = [acq_keys[i] for i in rng.permutation(len(acq_keys))]
            by = p.by_item()
            taken, seen = [], []
            m_a = zh.get((p.subject_id, p.benchmark_id), 0.0)
            for pos, k in enumerate(order):
                k_left, n_left = max_labels - len(taken), len(order) - pos
                if k_left <= 0:
                    break
                if k_left >= n_left:
                    taken.append(k); count[k] += 1; continue
                z = zb[kidx[k]]
                pz = sig(m_a + z)
                val = -count[k] + w_info * pz * (1 - pz)
                seen.append(val)
                q = 1.0 - min(0.999, k_left / n_left)
                take = len(seen) < warmup or val >= np.quantile(seen[-300:], max(0.0, q))
                if take:
                    taken.append(k); count[k] += 1
            new[(p.subject_id, p.benchmark_id)] = [(k, by[k][0].label) for k in taken]
    return new


def raw_eta(pairs, traj, zh, by_b, keys_of, emb, seed=0, use_prior=True):
    """Per-budget (benchmark, eta, floor, label) so calibration can be fitted
    on other benchmarks and applied to the held-out one."""
    out = {B: defaultdict(lambda: ([], [], [])) for B in E.BUDGETS}
    for B in E.BUDGETS:
        for bid, ps in by_b.items():
            kidx, X = keys_of[bid], emb[bid]
            zb = pooled_z(ps, traj, X, kidx, B)
            eta_l, c_l, y_l = out[B][bid]
            for p in ps:
                rows = traj[(p.subject_id, p.benchmark_id)][:B]
                Z = np.array([zb[kidx[k]] for k, _ in rows])
                Y = np.array([v for _, v in rows], float)
                m_a = zh.get((p.subject_id, p.benchmark_id), 0.0) if use_prior else 0.0
                a, b, va, vb = fit_ab(Z, Y, m_a, v_a, 1.0, v_b)
                _, ev = E.split_pair(p, seed); evs = set(ev)
                for r in p.responses:
                    if r.item_key not in evs:
                        continue
                    z = zb[kidx[r.item_key]]
                    k = 1 / np.sqrt(1 + (np.pi / 8) * (va + vb * z * z))
                    eta_l.append(k * (a + b * z))
                    c_l.append(floor_of(r.item["item_content"] or ""))
                    y_l.append(r.label)
    return out


def brier_from(eta, c, y, t=1.0, slip=0.02):
    eta, c, y = np.asarray(eta, float), np.asarray(c, float), np.asarray(y, float)
    p = np.clip(c + (1 - c - slip) * sig(t * eta), 1e-4, 1 - 1e-4)
    return float(np.mean((p - y) ** 2))
