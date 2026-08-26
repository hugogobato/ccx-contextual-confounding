"""Incumbent baselines for WP 2.3 (equal information, same data).

- inflation_order2_feasible : order-2 inflation LP for the IV class. Per
  formalization memo Prop 2, for shared-mechanism IV with no cross-copy
  observables the order-2 inflated system reduces EXACTLY to the Balke-Pearl
  equality system; this wrapper implements the inflated system faithfully
  (equality-form feasibility, independent of the slack LP implementation) and
  serves as the "inflation-LP test" incumbent. NOTE (deviation recorded for
  Gate B): github.com/ecboghiu/inflation v2.0.3 rejects the plain IV DAG
  ("directed edges between observed nodes lacking a common latent parent"),
  so the reference package cannot encode the scoped scenario without model
  surgery; spot-checks against it are therefore impossible for A1 and the
  wrapper stands in, cross-validated against the slack LP decisions.
- facet_battery             : max violation of a validated inequality system
  (exact sharp facets for binary IV; mechanically-derived valid inequalities
  elsewhere). Stand-in for the Pearl/Bonet/Kedagni-Mourifie closed-form
  battery family; every entry is machine-validated against the polytope.
- gtest_independence        : pooled G-test of X indep Y (ignores Z).
"""
import time

import numpy as np
from scipy.optimize import linprog
from scipy.stats import chi2

from models import discover_facets


# ------------------------------------------------------- order-2 inflation LP

def inflation_order2_feasible(M, cond, tol=1e-7):
    """Feasibility of exists tau in simplex: M tau == cond (equality form,
    HiGHS primal; post-checked residual). Returns (feasible, seconds)."""
    t0 = time.perf_counter()
    K, Q = M.shape
    res = linprog(np.zeros(Q),
                  A_eq=np.vstack([M.astype(float),
                                  np.ones((1, Q))]),
                  b_eq=np.concatenate([np.asarray(cond, float), [1.0]]),
                  bounds=(0, None), method="highs")
    dt = time.perf_counter() - t0
    if not res.success:
        return False, dt
    resid = float(np.max(np.abs(M.astype(float) @ res.x -
                                np.asarray(cond, float))))
    return bool(resid <= tol), dt


def inflation_slack_value(M, cond):
    """L1 distance to the feasible set (same information as order-2
    feasibility, used as the calibrated LP test statistic)."""
    K, Q = M.shape
    c = np.concatenate([np.zeros(Q), np.ones(K)])
    A_ub = np.zeros((2 * K, Q + K))
    A_ub[:K, :Q] = M
    A_ub[:K, Q:] = -np.eye(K)
    A_ub[K:, :Q] = -M
    A_ub[K:, Q:] = -np.eye(K)
    b_ub = np.concatenate([cond, -cond])
    A_eq = np.concatenate([np.ones(Q), np.zeros(K)])[None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if not res.success:
        return np.inf
    return float(max(res.fun, 0.0))


# ------------------------------------------------------------- facet battery

def build_exact_facets(M, cache=None, seed=0):
    """Exact facet system via the Phase 1 pipeline; returns (H, b) or None."""
    try:
        H, b = discover_facets(M, seed=seed)
        return H.astype(float), b.astype(float)
    except Exception:
        return None


def _lp_polytope_max(w, M):
    """max w . e s.t. e = M q, q in simplex (variables: q)."""
    Mf = np.asarray(M, float)
    w = np.asarray(w, float)
    Q = Mf.shape[1]
    res = linprog(-(Mf.T @ w),
                  A_eq=np.ones((1, Q)), b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if not res.success:
        return np.nan
    return float(w @ (Mf @ res.x))


def build_gm_battery(M, rng_seed=424242, n_candidates=4000,
                     support_max=6, keep=40, verbose=False):
    """Mechanically-derived valid inequality battery (stand-in for the
    Pearl/Bonet/KM closed-form family at arbitrary alphabets). Candidates:
    sparse +/-1 functionals of the conditional coordinates; each kept entry
    satisfies max_{polytope} w.e <= c (machine-validated by LP), i.e., the
    battery can NEVER fire on any law compatible with the structural class.
    Selection among valid candidates favors small c relative to spread on a
    random cloud (most violation-prone first). Returns list of (w, c)."""
    rng = np.random.default_rng(rng_seed)
    K, Q = M.shape
    W = np.zeros((n_candidates, K))
    for i in range(n_candidates):
        s = min(support_max, K)
        idx = rng.choice(K, size=rng.integers(2, s + 1), replace=False)
        W[i, idx] = rng.choice([-1.0, 1.0], size=len(idx))
    entries = []
    for i in range(n_candidates):
        w = W[i]
        if not np.any(w):
            continue
        c = _lp_polytope_max(w, M)
        if not np.isfinite(c):
            continue
        # informativeness: margin between polytope max and generic-cloud max
        entries.append((w, c))
    # rank by how tight the constant is relative to functional scale
    scored = []
    probe = rng.dirichlet(np.ones(Q), size=64) @ M.astype(float).T
    for w, c in entries:
        vals = probe @ w
        tightness = c - vals.max()  # >= 0; smaller => sharper on cloud
        scored.append((tightness, w, c))
    scored.sort(key=lambda t: t[0])
    battery = [(w, c) for _, w, c in scored[:keep]]
    if verbose:
        print(f"gm battery: {len(entries)} valid candidates, kept {len(battery)}")
    return battery


def facet_battery_stat(battery, cond):
    """Max violation of validated inequalities at the empirical conditionals.
    Negative-safe: max(0, max_j (w_j.e - c_j))."""
    e = np.asarray(cond, float)
    worst = -np.inf
    for w, c in battery:
        worst = max(worst, float(w @ e - c))
    return max(0.0, worst)


def pearl_facet_stat_binary(Hb, cond):
    """Binary IV: exact sharp facet system (H, b) from Phase 1; raw max
    violation."""
    v = float(np.max(np.asarray(Hb[0]) @ np.asarray(cond, float) -
                     np.asarray(Hb[1])))
    return max(0.0, v)


# ------------------------------------------------------------------ g-test

def gtest_independence_stat(counts, kz, kx, ky):
    """Pooled G statistic for X indep Y (contexts pooled; ignores Z)."""
    C = np.asarray(counts, dtype=float).reshape(kz, kx, ky)
    tab = C.sum(axis=0)
    n = tab.sum()
    if n <= 0:
        return 0.0
    row = tab.sum(axis=1, keepdims=True)
    col = tab.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(tab > 0,
                         tab * np.log(tab * n / (row * col)), 0.0)
    return float(2.0 * terms.sum())


def gtest_pvalue(stat, kx, ky):
    return float(chi2.sf(max(stat, 0.0), (kx - 1) * (ky - 1)))
