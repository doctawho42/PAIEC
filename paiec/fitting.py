"""Everything that survived, in one predictor.

  p = c + (1 - c - slip) * sigmoid(kappa * (a + b*z))

  c    guessing floor read off the item text (1/n options), 0 when not MCQ
  z    item difficulty: from the pooled joint IRT when labels pool across
       subjects, from the low-dimensional text features when they do not
  a    the subject's standing, with a prior from its attributes (relative only,
       the benchmark's level does not transfer and is left to the labels)
  b    how steeply this subject responds to difficulty, fitted per pair
  kappa Laplace shrinkage over the posterior of (a, b)
"""
import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))


def fit_ab(Z, Y, m_a=0.0, v_a=2.0, m_b=1.0, v_b=0.25, iters=40):
    a, b = m_a, m_b
    if len(Z) == 0:
        return a, b, v_a, v_b
    for _ in range(iters):
        p = np.clip(sig(a + b * Z), 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        g = np.array([np.sum(Y - p) - (a - m_a) / v_a,
                      np.sum((Y - p) * Z) - (b - m_b) / v_b])
        H = np.array([[np.sum(w) + 1 / v_a, np.sum(w * Z)],
                      [np.sum(w * Z), np.sum(w * Z * Z) + 1 / v_b]])
        s = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a += s[0]; b += s[1]
        if np.max(np.abs(s)) < 1e-10:
            break
    p = np.clip(sig(a + b * Z), 1e-6, 1 - 1e-6)
    w = p * (1 - p)
    H = np.array([[np.sum(w) + 1 / v_a, np.sum(w * Z)],
                  [np.sum(w * Z), np.sum(w * Z * Z) + 1 / v_b]])
    C = np.linalg.inv(H + 1e-9 * np.eye(2))
    return float(a), float(b), float(C[0, 0]), float(C[1, 1])
