"""CCX witnesses (definitions locked in memos/formalization_memo.md section 5).

All functionals take an observable vector e and a response map M (K x Q int)
whose columns are the images of deterministic response assignments; q ranges
over the Q-simplex.

- slack_and_feasible: L1 distance to the feasible polytope via LP; feasibility
  decision at tolerance.
- cf1_soft: probabilistic fraction t* = max sum(w) s.t. M w <= e, w >= 0.
- degree_signed: CbD-convention signed quasi-coupling TV distance.
- kl_contextuality: min_q conditional-KL of e from pushforward of q.
"""
import numpy as np
from scipy.optimize import linprog

import models as _models

TOL = 1e-9


def slack_and_feasible(M, e):
    """min_q in simplex ||M q - e||_1. Returns (value, feasible_flag)."""
    K, Q = M.shape
    c = np.concatenate([np.zeros(Q), np.ones(K)])
    A_ub = np.zeros((2 * K, Q + K))
    A_ub[:K, :Q] = M
    A_ub[:K, Q:] = -np.eye(K)
    A_ub[K:, :Q] = -M
    A_ub[K:, Q:] = -np.eye(K)
    b_ub = np.concatenate([e, -e])
    A_eq = np.concatenate([np.ones(Q), np.zeros(K)])[None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if not res.success:
        return np.inf, False
    val = max(res.fun, 0.0)
    return float(val), bool(val <= TOL)


def cf1_soft(M, e):
    """max sum w s.t. M w <= e, w >= 0. Value in [0,1]."""
    Q = M.shape[1]
    res = linprog(-np.ones(Q), A_ub=M.astype(float),
                  b_ub=e, bounds=(0, None), method="highs")
    if not res.success:
        return np.nan
    return float(max(res.fun * -1.0, 0.0))


def degree_signed(M, e):
    """(1/2) min over signed q with sum(q)=1 of ||M q - e||_1."""
    K, Q = M.shape
    c = np.concatenate([np.zeros(Q), np.ones(K)])
    A_ub = np.zeros((2 * K, Q + K))
    A_ub[:K, :Q] = M
    A_ub[:K, Q:] = -np.eye(K)
    A_ub[K:, :Q] = -M
    A_ub[K:, Q:] = -np.eye(K)
    b_ub = np.concatenate([e, -e])
    A_eq = np.concatenate([np.ones(Q), np.zeros(K)])[None, :]
    bounds = [(None, None)] * Q + [(0, None)] * K
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    if not res.success:
        return np.nan
    return float(max(res.fun, 0.0)) / 2.0


def kl_em_batch(M, E, n_iter=500, tol=1e-13, eps=1e-300):
    """Batched EM (multiplicative) updates for min_q in simplex KL(e || Mq).

    The objective is convex in q, so the fixed-point EM update converges to the
    global optimum from any positive start (uniform). Deterministic. Returns
    (kl_values (n,), q_star (n, Q))."""
    Mf = M.astype(float)
    n, Q = len(E), M.shape[1]
    q = np.full((n, Q), 1.0 / Q)
    prev = None
    for it in range(n_iter):
        pq = np.maximum(q @ Mf.T, eps)
        q *= (E / pq) @ Mf
        q /= q.sum(axis=1, keepdims=True)
        if it % 25 == 24 or it == n_iter - 1:
            pq = np.maximum(q @ Mf.T, eps)
            obj = -(E * np.log(pq)).sum(axis=1)
            if prev is not None:
                rel = float(np.max(np.abs(obj - prev) /
                                   np.maximum(1.0, np.abs(prev))))
                if rel <= tol:
                    break
            prev = obj
    pq = np.maximum(q @ Mf.T, eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(E > 0, E * (np.log(np.maximum(E, eps)) - np.log(pq)),
                        0.0)
    return np.maximum(term.sum(axis=1), 0.0), q


def kl_contextuality(M, e, **kwargs):
    """Single-instance wrapper of kl_em_batch."""
    val, _ = kl_em_batch(M, e[None, :], **kwargs)
    return float(val[0])


# ------------------------------------------------- S_IV-specific hierarchy LPs

def maximal_support_lp(A_z, supp_mask):
    """max m s.t. pushforward >= m on every support row of context z, q simplex."""
    K, Q = A_z.shape
    idx = np.where(supp_mask)[0]
    c = np.zeros(Q + 1)
    c[-1] = -1.0
    A_ub = np.zeros((len(idx), Q + 1))
    A_ub[:, :Q] = -A_z[idx]
    A_ub[:, Q] = 1.0
    b_ub = np.zeros(len(idx))
    A_eq = np.concatenate([np.ones(Q), np.zeros(1)])[None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * Q + [(None, None)], method="highs")
    if not res.success:
        return np.nan
    return float(res.fun * -1.0)


def iv_cf1_hard(e, A=None):
    """Max fraction of contexts whose support contains pi_z(theta), theta the 16
    deterministic tables."""
    if A is None:
        A = _models.build_iv_A()

    tabs_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tabs_y = [(0, 0), (0, 1), (1, 0), (1, 1)]
    best = 0
    for ix, tx in enumerate(tabs_x):
        for iy, ty in enumerate(tabs_y):
            ok = 0
            for z in (0, 1):
                x = tx[z]
                y = ty[x]
                ok += int(e[4 * z + 2 * x + y] > 0)
            best = max(best, ok)
    return best / 2.0


def iv_inflation2_feasible(e, A=None, p_floor=None):
    """Order-2 cross-check per memo Prop 2. Copies share the mechanism, hence
    the same response-table law tau, and observational IV data provides no
    cross-copy observables; the inflated feasibility question therefore reduces
    EXACTLY to the Balke-Pearl system exists tau: A_z tau = e_z for both z.
    Implemented here independently of slack_and_feasible (pure equality-form
    feasibility rather than L1-slack minimization) to serve as a solver-level
    cross-validator on a subsample."""
    if A is None:
        A = _models.build_iv_A()

    res = linprog(np.zeros(16), A_ub=-A.astype(float), b_ub=np.zeros(8),
                  A_eq=np.vstack([A, np.ones((1, 16))]).astype(float),
                  b_eq=np.concatenate([e, [1.0]]), bounds=(0, None),
                  method="highs")
    if not res.success:
        return False
    resid = float(np.max(np.abs(A.astype(float) @ res.x - e)))
    return resid <= 1e-7
