"""WP 2.2 calibration machinery: null fitting, resampling engines, critical
values.

Engines (all operate on counts; fair comparison: the SAME replicate index b
indexes identically-shaped draws across statistics):
- para_boot : parametric bootstrap under the fitted null; strata sizes drawn
              Multinomial(n, fitted p_z), within-stratum Multinomial(n_z*, e*).
- crt_cond  : conditional randomization test form: observed strata sizes n_z
              held fixed (conditioning on Z counts); within-stratum draws from
              the fitted null conditionals.
- subsample : sampling without replacement from the OBSERVED counts at size
              m = n^0.7 (multivariate hypergeometric per stratum).

Fitted null = KL projection q_hat of the observed conditionals onto the
response polytope (EM, deterministic). The same q_hat feeds all engines.
"""
import numpy as np

from witnesses import kl_em_batch

ENGINES = ("para_boot", "crt_cond", "subsample")


def fit_null_q(M, cond, n_iter=800, tol=1e-12):
    """KL projection onto {M q : q in simplex}; returns (q_hat (Q,), push
    (kz, kx*ky)). cond: (K,) conditionals."""
    _, q = kl_em_batch(M.astype(float), np.asarray(cond, float)[None, :],
                       n_iter=n_iter, tol=tol)
    q_hat = q[0]
    push = M.astype(float) @ q_hat
    return q_hat, push


def draw_bootstrap_counts(push, obs_counts, kz, rng, B, engine,
                          subsample_exponent=0.7):
    """Return (B, K) count matrices for the chosen engine.

    push      : (kz, kx*ky) fitted-null conditionals per stratum.
    obs_counts: (K,) observed counts.
    """
    C = np.asarray(obs_counts, dtype=np.int64)
    K = len(C)
    blk = K // kz
    n_z = np.array([C[blk * z:blk * (z + 1)].sum() for z in range(kz)])
    P = np.maximum(np.asarray(push, float), 0.0)
    P = P / P.sum(axis=1, keepdims=True)

    out = np.zeros((B, K), dtype=np.int64)
    if engine == "crt_cond":
        for z in range(kz):
            s = slice(blk * z, blk * (z + 1))
            if n_z[z] > 0:
                out[:, s] = rng.multinomial(int(n_z[z]), P[z], size=B)
        return out
    if engine == "para_boot":
        n_tot = int(n_z.sum())
        p_z = n_z / max(n_z.sum(), 1)
        nz_star = rng.multinomial(n_tot, p_z, size=B)  # (B, kz)
        for z in range(kz):
            s = slice(blk * z, blk * (z + 1))
            nz_col = nz_star[:, z]
            for nz_val in np.unique(nz_col):
                if nz_val == 0:
                    continue
                rows = np.where(nz_col == nz_val)[0]
                drawn = rng.multinomial(int(nz_val), P[z],
                                        size=len(rows))
                out[np.ix_(rows, np.arange(blk * z, blk * (z + 1)))] = drawn
        return out
    if engine == "subsample":
        n_tot = int(n_z.sum())
        m = max(2, min(int(np.floor(n_tot ** subsample_exponent)), n_tot))
        mz = _alloc_strata(n_z, m)
        for z in range(kz):
            s = slice(blk * z, blk * (z + 1))
            if mz[z] > 0:
                for b in range(B):
                    out[b, s] = rng.multivariate_hypergeometric(
                        C[s], int(mz[z]))
        return out
    raise ValueError(f"unknown engine {engine}; expected one of {ENGINES}")


def _alloc_strata(n_z, m):
    """Allocate m total draws across strata proportionally to observed stratum
    shares, capped by stratum sizes; remainder distributed to residual room."""
    share = n_z / max(n_z.sum(), 1)
    mz = np.minimum(np.floor(m * share), n_z).astype(np.int64)
    rem = m - int(mz.sum())
    order = np.argsort(-(n_z - mz))
    while rem > 0:
        moved = False
        for zi in order:
            add = min(rem, int(n_z[zi]) - int(mz[zi]))
            if add > 0:
                mz[zi] += add
                rem -= add
                moved = True
            if rem == 0:
                break
        if not moved:
            break
    return mz


def critical_values(stat_draws, alpha_grid):
    """Upper-tail critical values: (1-alpha) empirical quantiles of bootstrap
    draws under the resampled null."""
    s = np.sort(np.asarray(stat_draws, dtype=float))
    out = {}
    for a in alpha_grid:
        idx = min(max(int(np.ceil((1.0 - a) * len(s))) - 1, 0), len(s) - 1)
        out[float(a)] = float(s[idx])
    return out


def pooled_critical_value(draw_lists, alpha):
    """Pooled quantile over several bootstrap-draw collections (used when
    critical values are estimated from multiple null datasets)."""
    s = np.sort(np.concatenate([np.asarray(d, float) for d in draw_lists]))
    idx = min(max(int(np.ceil((1.0 - alpha) * len(s))) - 1, 0), len(s) - 1)
    return float(s[idx])
