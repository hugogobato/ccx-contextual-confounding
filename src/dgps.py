"""CCX seeded samplers (Phase 1 streams; see configs/seeds.json)."""
import numpy as np


def uniform_conditionals(rng, n):
    """Dirichlet(1) per context: returns (n, 8) conditionals, z-major blocks."""
    a = rng.dirichlet(np.ones(4), size=n)
    b = rng.dirichlet(np.ones(4), size=n)
    return np.concatenate([a, b], axis=1)


def project_direction(d):
    """Project a direction on observable space onto the affine hull
    (per-context block sums zero) so rays stay comparable across contexts."""
    d = d.reshape(-1)
    out = d.copy()
    out[0:4] -= d[0:4].mean()
    out[4:8] -= d[4:8].mean()
    return out


def ray_exit_t(H, b, e_f, d):
    """Max t >= 0 with H(e_f + t d) <= b (closed form); None if unbounded."""
    Hd = H.astype(float) @ d
    he = H.astype(float) @ e_f - b.astype(float)  # <= 0 inside
    with np.errstate(divide="ignore"):
        cand = np.where(Hd > 1e-15, -he / np.where(Hd > 1e-15, Hd, 1.0),
                        np.inf)
    tmax = float(cand.min())
    return None if not np.isfinite(tmax) else max(tmax, 0.0)


def sparse_conditionals(rng, n, mix=((1, 0.4), (2, 0.4), (3, 0.2))):
    """Sparse-support instances: Dirichlet draws with entries zeroed per
    context block (renormalized). Mix gives fractions of 1-, 2-, 3-zero blocks;
    support sizes 3/2/1. Support-size-1 blocks can produce strongly contextual
    instances (cf. micro-example B)."""
    ks = np.array([k for k, _ in mix])
    ps = np.array([w for _, w in mix])
    ps = ps / ps.sum()
    out = np.empty((n, 8))
    for i in range(n):
        blocks = []
        for z in (0, 1):
            v = rng.dirichlet(np.ones(4))
            k = int(rng.choice(ks, p=ps))
            idx = rng.choice(4, size=k, replace=False)
            v[idx] = 0.0
            blocks.append(v / v.sum())
        out[i] = np.concatenate(blocks)
    return out


def boundary_batch(A, H, b, n, seed, rho_grid=(0.90, 0.95, 0.99, 1.005, 1.01,
                                               1.02, 1.05)):
    """Near-facet stream: feasible points advanced along affine directions,
    placed at rho * t_exit for log-spaced rho around the boundary."""
    rng = np.random.default_rng(seed)
    Q = A.shape[1]
    out = []
    while len(out) < n:
        m = min(256, n - len(out))
        qf = rng.dirichlet(np.ones(Q), size=m)
        ef = qf @ A.T.astype(float)
        ds = np.array([project_direction(x) for x in
                       rng.normal(size=(m, 8))])
        norms = np.linalg.norm(ds, axis=1, keepdims=True)
        ds = ds / np.maximum(norms, 1e-12)
        for i in range(m):
            t_exit = ray_exit_t(H, b, ef[i], ds[i])
            if t_exit is None or t_exit <= 0:
                continue
            rho = float(rng.choice(rho_grid))
            out.append(ef[i] + rho * t_exit * ds[i])
            if len(out) == n:
                break
    return np.array(out)
