"""WP 2.1 witness estimators (Phase 2).

Statistics consume per-dataset COUNTS n(z, x, y) (flattened row vector,
row index = kx*ky*z + ky*x + y, matching build_iv_A_general rows) and return
scalar statistics. All estimators are functions of the observable counts only.

Estimator variants (plan WP 2.1):
- plugin        : witness evaluated on full-sample conditional frequencies.
- split         : fit the coupling on one half of the sample (per-stratum
                  hypergeometric split), evaluate the witness on the held-out
                  half with the fitted coupling frozen.
- crossfit (debiased): 2-fold cross-fitting; average of both held-out
                  directions. Kills the plugin's downward min-bias without
                  losing half the data on average.

Witness families: kl (primary, Gate A decision), cf1 (effect-size companion,
margin 1 - t*), slack (L1 distance; population-equivalent to order-2
inflation feasibility per formalization memo Prop 2).

Population values reproduce Phase 1 witnesses exactly when fed the exact
population conditionals (unit-tested at tolerance 1e-8).
"""
import numpy as np
from scipy.optimize import linprog

from witnesses import kl_em_batch

TOL = 1e-9


# ------------------------------------------------------------- count plumbing

def counts_to_conditionals(counts, kz):
    """Row-stacked conditional blocks e_z from a batch of count vectors.
    counts: (B, K) ints/floats with zero strata allowed. Returns (B, K)
    float conditionals; empty strata produce uniform placeholders (never used:
    KL terms with E=0 contribute nothing; cf1/slack LPs get b=0 blocks which
    are trivially satisfiable by q mass placement... guarded upstream by
    requiring p_z in [0.1, 0.9] at DGP level)."""
    C = np.asarray(counts, dtype=float)
    B = C.shape[0]
    K = C.shape[1]
    out = np.empty_like(C)
    blk = K // kz
    for z in range(kz):
        s = slice(blk * z, blk * (z + 1))
        tot = C[:, s].sum(axis=1, keepdims=True)
        safe = np.where(tot > 0, tot, 1.0)
        out[:, s] = C[:, s] / safe
    return out


def split_counts(counts, kz, rng, frac):
    """Single-dataset split: counts (K,) -> (fit-part, eval-part)."""
    ca, cb = _split_counts_batch(np.asarray(counts, dtype=np.int64)[None, :],
                                 kz, rng, frac)
    return ca[0], cb[0]


def _split_counts_batch(counts, kz, rng, frac):
    out_a = np.zeros_like(counts)
    K = counts.shape[1]
    blk = K // kz
    for z in range(kz):
        s = slice(blk * z, blk * (z + 1))
        nz = counts[:, s].sum(axis=1)
        for i in range(counts.shape[0]):
            if nz[i] == 0:
                continue
            m = int(round(frac * nz[i]))
            m = max(1, min(int(nz[i]) - 1, m)) if nz[i] >= 2 else int(nz[i])
            a = rng.multivariate_hypergeometric(counts[i, s], m)
            out_a[i, s] = a
    return out_a, counts - out_a


# ------------------------------------------------------------------ KL family

def kl_plugin_stat(M, counts, kz):
    """min_q sum_z KL(e_hat_z || pi_z(q)); EM to convergence (deterministic)."""
    E = counts_to_conditionals(counts[None, :], kz)
    val, _ = kl_em_batch(M.astype(float), E)
    return float(val[0])


def _kl_fit_q(M, E_fit):
    _, q = kl_em_batch(M.astype(float), E_fit)
    return q


def kl_split_stat(M, counts, kz, rng, frac=0.5):
    """Fit q on part A (EM), evaluate sum_z KL(e_B_z || pi_z(q_A))."""
    c = np.asarray(counts, dtype=np.int64)[None, :]
    ca, cb = _split_counts_batch(c, kz, rng, frac)
    Ea = counts_to_conditionals(ca, kz)
    Eb = counts_to_conditionals(cb, kz)
    qa = _kl_fit_q(M, Ea)
    return float(_kl_eval_with_q(M, Eb, qa)[0])


def _kl_eval_with_q(M, E, q):
    pq = np.maximum(q @ M.astype(float).T, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(E > 0, E * (np.log(np.maximum(E, 1e-300)) -
                                    np.log(pq)), 0.0)
    return np.maximum(term.sum(axis=1), 0.0)


def kl_crossfit_stat(M, counts, kz, rng, folds=2, frac=None):
    """Debiased cross-fitted KL: average over fold pairs of held-out KL at
    fold-fitted couplings."""
    c = np.asarray(counts, dtype=np.int64)[None, :]
    vals = []
    idx_pairs = []
    for _ in range(folds):
        ca, cb = _split_counts_batch(c, kz, rng, 0.5)
        idx_pairs.append((ca, cb))
    tot = [(a, b) for a, b in idx_pairs]
    # cross-fit: direction 1 uses fold1 fit -> fold2 eval and vice versa
    (c1, c2), (c3, c4) = tot[0], tot[min(1, len(tot) - 1)]
    Ea1, Eb1 = counts_to_conditionals(c1, kz), counts_to_conditionals(c2, kz)
    Ea2, Eb2 = counts_to_conditionals(c3, kz), counts_to_conditionals(c4, kz)
    v12 = _kl_eval_with_q(M, Eb1, _kl_fit_q(M, Ea1))[0]
    v21 = _kl_eval_with_q(M, Ea2, _kl_fit_q(M, Eb2))[0]
    vals = [float(v12), float(v21)]
    return float(np.mean(vals))


def kl_bootstrap_batch(M, cond_draws):
    """Batched plugin KL over bootstrap draws. cond_draws: (B, K)."""
    val, _ = kl_em_batch(M.astype(float), cond_draws)
    return val


# ----------------------------------------------------------------- cf1 family

def cf1_plugin_stat(M, counts, kz):
    """t*(e_hat) via LP; statistic reported as margin 1 - t*. Value = t*."""
    E = counts_to_conditionals(counts[None, :], kz)[0]
    Q = M.shape[1]
    res = linprog(-np.ones(Q), A_ub=M.astype(float), b_ub=E,
                  bounds=(0, None), method="highs")
    if not res.success:
        return np.nan
    return float(max(res.fun * -1.0, 0.0))


def cf1_split_stat(M, counts, kz, rng, frac=0.5):
    """Fit w on part A; frozen-w coverage ratio on part B capped at 1:
    t_frozen = min_{o: (M w)_o > 0} e_B,o / (M w)_o."""
    c = np.asarray(counts, dtype=np.int64)[None, :]
    ca, cb = _split_counts_batch(c, kz, rng, frac)
    Ea = counts_to_conditionals(ca, kz)[0]
    Eb = counts_to_conditionals(cb, kz)[0]
    Q = M.shape[1]
    res = linprog(-np.ones(Q), A_ub=M.astype(float), b_ub=Ea,
                  bounds=(0, None), method="highs")
    if not res.success:
        return np.nan
    w = np.maximum(res.x, 0.0)
    cov = M.astype(float) @ w
    mask = cov > 1e-15
    if not np.any(mask):
        return 0.0
    return float(min(1.0, np.min(Eb[mask] / cov[mask])))


# --------------------------------------------------------------- slack family

def slack_plugin_stat(M, counts, kz):
    """min_q ||M q - e_hat||_1 (LP); the order-2-inflation-equivalent
    distance statistic (memo Prop 2)."""
    E = counts_to_conditionals(counts[None, :], kz)[0]
    K, Q = M.shape
    c = np.concatenate([np.zeros(Q), np.ones(K)])
    A_ub = np.zeros((2 * K, Q + K))
    A_ub[:K, :Q] = M
    A_ub[:K, Q:] = -np.eye(K)
    A_ub[K:, :Q] = -M
    A_ub[K:, Q:] = -np.eye(K)
    b_ub = np.concatenate([E, -E])
    A_eq = np.concatenate([np.ones(Q), np.zeros(K)])[None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if not res.success:
        return np.inf
    return float(max(res.fun, 0.0))


def slack_split_stat(M, counts, kz, rng, frac=0.5):
    """Fit q on part A (EM projection); L1 gap of held-out conditionals to
    the frozen pushforward M q_A."""
    c = np.asarray(counts, dtype=np.int64)[None, :]
    ca, cb = _split_counts_batch(c, kz, rng, frac)
    Ea = counts_to_conditionals(ca, kz)
    Eb = counts_to_conditionals(cb, kz)
    qa = _kl_fit_q(M, Ea)
    push = qa @ M.astype(float).T
    return float(np.abs(push - Eb).sum(axis=1)[0])


def kl_split_crossfit_stat(M, counts, kz, rng, frac=0.5):
    """Single-dataset split + crossfit sharing directional fits:
    returns (split=v_ab, crossfit=(v_ab+v_ba)/2)."""
    sp, cf = kl_split_crossfit_batch(M, np.asarray(counts,
                                                   dtype=np.int64)[None, :],
                                     kz, rng, chunk=1, frac=frac)
    return float(sp[0]), float(cf[0])


def kl_split_crossfit_batch(M, counts, kz, rng, chunk=32, frac=0.5,
                            fit_iter=150, fit_tol=1e-9):
    """Chunk-batched split + crossfit over a batch of bootstrap count
    vectors. counts: (Bc, K). Returns (split (Bc,), crossfit (Bc,)).
    Directional fits use capped EM iterations (quantile-level precision)."""
    C = np.asarray(counts, dtype=np.int64)
    Bc = C.shape[0]
    Mf = M.astype(float)
    sp_out = np.empty(Bc)
    cf_out = np.empty(Bc)
    for s in range(0, Bc, chunk):
        e = min(s + chunk, Bc)
        ca, cb = _split_counts_batch(C[s:e], kz, rng, frac)
        Ea = counts_to_conditionals(ca, kz)
        Eb = counts_to_conditionals(cb, kz)
        both = np.vstack([Ea, Eb])
        _, q = kl_em_batch(Mf, both, n_iter=fit_iter, tol=fit_tol)
        qa, qb = q[:e - s], q[e - s:]
        pa = np.maximum(qa @ Mf.T, 1e-300)
        pb = np.maximum(qb @ Mf.T, 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_ab = np.where(Eb > 0,
                            Eb * (np.log(np.maximum(Eb, 1e-300)) -
                                  np.log(pa)), 0.0).sum(axis=1)
            v_ba = np.where(Ea > 0,
                            Ea * (np.log(np.maximum(Ea, 1e-300)) -
                                  np.log(pb)), 0.0).sum(axis=1)
        sp_out[s:e] = np.maximum(v_ab, 0.0)
        cf_out[s:e] = np.maximum(0.5 * (v_ab + v_ba), 0.0)
    return sp_out, cf_out


STAT_REGISTRY = {
    "kl": {
        "plugin": lambda M, counts, kz, rng: kl_plugin_stat(M, counts, kz),
        "split": lambda M, counts, kz, rng:
            kl_split_crossfit_stat(M, counts, kz, rng)[0],
        "crossfit": lambda M, counts, kz, rng:
            kl_split_crossfit_stat(M, counts, kz, rng)[1],
    },
    "cf1": {
        "plugin": lambda M, counts, kz, rng: cf1_plugin_stat(M, counts, kz),
        "split": lambda M, counts, kz, rng: cf1_split_stat(M, counts, kz, rng),
    },
    "slack": {
        "plugin": lambda M, counts, kz, rng: slack_plugin_stat(M, counts, kz),
        "split": lambda M, counts, kz, rng:
            slack_split_stat(M, counts, kz, rng),
    },
}
