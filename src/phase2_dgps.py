"""Phase 2 seeded DGP library (WP 2.2 / 2.3).

All DGPs produce ROW-LEVEL datasets (n, 3) int columns (z, x, y); statistics
consume counts via counts_from_rows. Alphabets (kz, kx, ky) per cell.

Null kinds (all are IV-form SCMs, hence laws lie in P_S by construction):
- chain        : Z -> X -> Y only (no latent).
- inert_u      : latent U drawn but mechanism-independent (presence control).
- u_on_x       : U enters X's mechanism only.
- u_on_y       : U enters Y's mechanism only.
- anchor_deterministic : fixed identity-chain law (ME-A style, boundary point).
- anchor_interior      : fixed interior feasible law (ME-C style generalization).

Alternative families:
- mixture (primary, all cells): response-space mixture q_rho =
  (1-rho) q0 + rho q_alt with q0 product-form (feasible) and q_alt a fixed
  generic coupling chosen per cell so that the rho=1 law is contextual;
  strength rho = mass of the confounded component.
- mechanistic (secondary, binary cell): logistic-threshold shared-U SCM.

Contamination stress (WP 2.2): each row independently with prob eps replaced
by a uniform (x, y) draw within its observed context z (data corruption). The
contaminated POPULATION law is exposed for feasibility classification
(feasible-contamination cells count toward size; infeasible ones toward
detection power), per the predeclared protocol in memos/gate_B_memo.md.
"""
import numpy as np

from models import build_iv_A_general, _mixed_radix_tables
from witnesses import kl_em_batch


# ------------------------------------------------------------- mechanisms

def _rand_mech(rng, k_out, k_in):
    return rng.integers(0, k_out, size=k_out ** k_in)


def sample_null_mechanism(rng, kz, kx, ky, kind,     p_conc=None, p_z_given=None):
    """Return (samp(z,u) -> (x,y), p_z). Mechanisms are deterministic tables
    over (z, u); independent noise is absorbed by table randomness.

    p_z_given (Phase 4 A1): use this exact context mass (shared across
    mechanisms so that observable mixtures keep equal per-z weights and
    stay feasible at the conditional level).
    p_conc=None (default): interior-constrained p_z as in WP 2.2.
    p_conc=c (Phase 4 boundary-stress / tail=t3 arm): p_z ~ Dirichlet(c)
    with NO interior constraint (bursty context masses)."""
    if p_z_given is not None:
        p_z = np.asarray(p_z_given, float)
    elif p_conc is None:
        while True:
            p_z = rng.dirichlet(np.ones(kz))
            if np.all((p_z > 0.10) & (p_z < 0.90)):
                break
    else:
        p_z = rng.dirichlet(np.ones(kz) * p_conc)
    if kind == "chain":
        f_z = _rand_mech(rng, kx, 1)
        g_x = _rand_mech(rng, ky, 1)

        def samp(z, u):
            x = int(f_z[z])
            y = int(g_x[x])
            return x, y
    elif kind == "inert_u":
        f_z = _rand_mech(rng, kx, 1)
        g_x = _rand_mech(rng, ky, 1)

        def samp(z, u):
            x = int(f_z[z])
            return x, int(g_x[x])
    elif kind == "u_on_x":
        f_zu = _rand_mech(rng, kx, kz + 1)
        g_x = _rand_mech(rng, ky, 1)

        def samp(z, u):
            x = int(f_zu[z * 2 + u])
            y = int(g_x[x])
            return x, y
    elif kind == "u_on_y":
        f_z = _rand_mech(rng, kx, 1)
        g_xu = _rand_mech(rng, ky, kx + 1)

        def samp(z, u):
            x = int(f_z[z])
            y = int(g_xu[x * 2 + u])
            return x, y
    else:
        raise ValueError(kind)
    return samp, p_z


def sample_rows_null(rng, n, kz, kx, ky, kind, p_conc=None, p_z_given=None):
    """Fresh IV-form null SCM per replication; returns rows (n,3), meta with
    the POPULATION conditional blocks (mechanisms evaluated exactly)."""
    samp, p_z = sample_null_mechanism(rng, kz, kx, ky, kind, p_conc=p_conc,
                                      p_z_given=p_z_given)
    z = rng.choice(kz, size=n, p=p_z)
    u = rng.integers(0, 2, size=n)
    pairs = z * 2 + u
    uniq, inv = np.unique(pairs, return_inverse=True)
    outs = np.array([samp(int(pv // 2), int(pv % 2)) for pv in uniq])
    xy = np.asarray(outs)[inv]
    rows = np.stack([z, xy[:, 0], xy[:, 1]], axis=1)
    blk = kx * ky
    pop = np.zeros((kz, blk))
    for zv in range(kz):
        for uv in (0, 1):
            x, y = samp(zv, uv)
            pop[zv, x * ky + y] += 0.5 * p_z[zv]
    return rows, {"p_z": p_z.tolist(), "kind": kind, "pop_cond": pop}


def anchor_deterministic_rows(kz, kx, ky, n):
    """Identity chain law: x = z mod kx, y = x mod ky; balanced contexts."""
    per_z = max(n // kz, 1)
    zs, xs, ys = [], [], []
    for z in range(kz):
        x = z % kx
        ys.append(np.full(per_z, x % ky))
        xs.append(np.full(per_z, x))
        zs.append(np.full(per_z, z))
    return np.stack([np.concatenate(zs), np.concatenate(xs),
                     np.concatenate(ys)], axis=1)


def anchor_interior_conditional(M, kz, kx, ky, seed=1234):
    """Interior feasible conditional blocks from a concentrated random q."""
    rng = np.random.default_rng(seed)
    Q = M.shape[1]
    q = rng.dirichlet(np.ones(Q) * 3.0)
    push = M.astype(float) @ q
    blk = kx * ky
    cond = push.reshape(kz, blk)
    return cond / cond.sum(axis=1, keepdims=True)


def sample_rows_from_conditional(rng, n, cond, kz, kx, ky,
                                 share=None):
    """iid rows from per-context conditional blocks; context shares fixed
    uniform unless given (context mass is nuisance, not part of H0)."""
    blk = kx * ky
    if share is None:
        share = np.full(kz, 1.0 / kz)
    z = rng.choice(kz, size=n, p=share)
    xy_idx = np.empty(n, dtype=np.int64)
    for zz in range(kz):
        mask = z == zz
        nm = int(mask.sum())
        if nm:
            xy_idx[mask] = rng.choice(blk, size=nm, p=cond[zz])
    return np.stack([z, xy_idx // ky, xy_idx % ky], axis=1)


# ------------------------------------------------- mixture alternative family

class MixtureCell:
    """Per-cell precomputation for the predeclared strength family:
    e_rho = (1-rho) * e_0 + rho * e_C, where e_0 = pushforward of a
    product-form coupling (EXACTLY feasible) and e_C is a fixed canonical
    CONFLICT law (per-context point masses that no coupling realizes,
    found by seeded search and verified contextual via the exact KL
    functional). Mixing happens at the OBSERVABLE level: pushforwards of
    couplings are always feasible, so rho parametrizes distance outside
    the polytope (witness(e_rho) grows monotonically from 0)."""

    def __init__(self, kz, kx, ky, alt_seed=20260825, min_witness=0.05,
                 anchor_conc=1.0):
        """anchor_conc: Dirichlet concentration of the product-form anchor
        (e_0). Default 1.0 = WP 2.3 behavior; anchor_conc < 1 (Phase 4
        tail=t3 arm) yields near-vertex product anchors, i.e. boundary-
        stress alternatives that remain EXACTLY feasible at rho=0."""
        self.kz, self.kx, self.ky = kz, kx, ky
        self.M = build_iv_A_general(kz, kx, ky)
        Mf = self.M.astype(float)
        tabs_x = _mixed_radix_tables(kx, kz)
        tabs_y = _mixed_radix_tables(ky, kx)
        nt_y = len(tabs_y)
        nt_x = len(tabs_x)
        rng = np.random.default_rng(alt_seed)
        blk = kx * ky
        K = kz * blk

        # ---- feasible anchor e_0: product-form coupling pushforward
        mu = rng.dirichlet(np.ones(nt_x) * anchor_conc)
        nu = rng.dirichlet(np.ones(nt_y) * anchor_conc)
        q0 = np.zeros(self.M.shape[1])
        for ix in range(nt_x):
            for iy in range(nt_y):
                q0[nt_y * ix + iy] = mu[ix] * nu[iy]
        self.e0 = q0 @ Mf.T

        # ---- canonical conflict law e_C: search point-mass pairs
        def stacked_from_cells(cells):
            e = np.zeros(K)
            for z in range(kz):
                x, y = cells[z]
                e[z * blk + x * ky + y] = 1.0
            return e

        self.eC = None
        best_v = -1.0
        seen = set()
        for _attempt in range(400):
            cells = [(int(rng.integers(kx)), int(rng.integers(ky)))
                     for _ in range(kz)]
            key = tuple(cells)
            if key in seen:
                continue
            seen.add(key)
            e = stacked_from_cells(cells)
            val, _ = kl_em_batch(Mf, e[None, :])
            if float(val[0]) > best_v:
                best_v = float(val[0])
                self.eC = e
                self.eC_cells = cells
            if best_v >= min_witness:
                break
        assert self.eC is not None and best_v > 0, \
            f"no conflict law found for cell {(kz, kx, ky)}"

    def population_conditional(self, rho):
        e = (1.0 - rho) * self.e0 + rho * self.eC
        blk = self.kx * self.ky
        cond = e.reshape(self.kz, blk)
        return cond / cond.sum(axis=1, keepdims=True)

    def sample_rows(self, rng, n, rho):
        return sample_rows_from_conditional(rng, n,
                                            self.population_conditional(rho),
                                            self.kz, self.kx, self.ky)


# ------------------------------------------------- mechanistic binary family

def mechanistic_binary_rows(rng, n, rho, params_seed=900001):
    """Logistic-threshold IV SCM with shared U: P(X=1|z,u) = sigmoid(a_z +
    rho*s_x*u), P(Y=1|x,u) = sigmoid(b_x + rho*s_y*u). Coefficients fixed
    across replications (drawn once from params_seed); relevance enforced."""
    prng = np.random.default_rng(params_seed)
    a = prng.normal(scale=1.0, size=2)
    while abs(a[1] - a[0]) < 0.5:
        a = prng.normal(scale=1.0, size=2)
    b = prng.normal(scale=1.0, size=2)
    s_x, s_y = 2.0, 2.0

    def sig(t):
        return 1.0 / (1.0 + np.exp(-t))

    z = rng.integers(0, 2, size=n)
    u = rng.integers(0, 2, size=n)
    px = sig(a[z] + rho * s_x * u)
    x = (rng.random(n) < px).astype(np.int64)
    py = sig(b[x] + rho * s_y * u)
    y = (rng.random(n) < py).astype(np.int64)
    return np.stack([z, x, y], axis=1)


# -------------------------------------------------------------- contamination

def contaminate_rows(rng, rows, kx, ky, eps):
    """With prob eps replace (x, y) by an independent uniform draw within the
    row's observed context z (gross data corruption)."""
    rows = rows.copy()
    hit = rng.random(len(rows)) < eps
    nh = int(hit.sum())
    if nh:
        rows[hit, 1] = rng.integers(0, kx, size=nh)
        rows[hit, 2] = rng.integers(0, ky, size=nh)
    return rows


def contaminated_population_conditional(cond_pop, eps):
    blk = cond_pop.shape[1]
    uni = np.full(blk, 1.0 / blk)
    return (1 - eps) * np.asarray(cond_pop, float) + eps * uni


def counts_from_rows(rows, kz, kx, ky):
    K = kz * kx * ky
    flat = (rows[:, 0] * kx + rows[:, 1]) * ky + rows[:, 2]
    return np.bincount(flat, minlength=K).astype(np.int64)
