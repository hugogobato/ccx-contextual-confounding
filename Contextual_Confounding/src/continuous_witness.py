"""WP 3.1 continuous-outcome witnesses (Phase 3 flagship, claim C4).

Constructions (predeclared design; see memos/gate_C_memo.md):

Context cover: X-strata (quantile bins of the treatment/residual covariate),
K strata by n (K = 3..8). Within each context c, ADDITIVE-MECHANISM
residuals r = y - m_hat(x) form an empirical law mu_c.

Declared structural null S_cont (homoskedastic shared-shape ANM):
y = m(x) + sigma * eps, eps ~ F shared, eps independent of x. After
per-context affine standardization (center/scale by context residual
moments), every context law must COINCIDE with one common shape. Latent
common causes break this: U enters both arms, so stratum-restricted U
ranges inject stratum-dependent variance/skew into standardized residuals.

K1 (kernelized forcing-divergence witness):
  T_K1 = min_{w in simplex} sum_c MMD_k^2( mu_hat_c , nu_c(w) ),
where nu_c(w) is the Gaussian-smoothed mixture with weights w over a fixed
component grid t_1..t_G (standard-normal quantiles) pushed through the
context affine map. Mean embeddings are linear in w => exact QP per sum;
solved by projected gradient (Frank-Wolfe) to machine tolerance. Value 0
iff ONE shared coupling explains all context laws simultaneously (in the
kernel-smoothed sense) - the continuous descendant of the discrete KL
contextuality functional (KL -> MMD^2 forcing divergence).

K2 (transport-defect witness):
  Consecutive-context squared Wasserstein distances d_c =
  W2^2(mu_c, mu_{c+1}) (exact 1-D empirical OT). Under the shared-shape
  null with drift, d_c ~ (m_{c+1} - m_c)^2 up to sampling error; the
  witness is the max standardized EXCESS transport
  T_K2 = max_c (d_c - fitted)/se, the cycle-space/monodromy-excess
  descendant of B2's topological statistic, requiring no support holes.

Calibration: multiplier (Rademacher wild) bootstrap recomputing the whole
pipeline on multiplier-perturbed residuals, including the inner
optimization; optional trimming policy applied BEFORE context moments
(predeclared grids q in {0, 0.01, 0.05}).

All estimators take arrays (x, y) and return scalar statistics; bootstrap
variants return draws for critical values.
"""
import numpy as np

try:
    import ot as pot
except Exception:                                   # pragma: no cover
    pot = None


# ------------------------------------------------------------------ contexts

def make_contexts(x, K=None, return_edges=False):
    """Quantile-stratify x into K contexts. Default K by n: 3 (<800),
    5 (<4000), 8 otherwise (predeclared)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if K is None:
        K = 3 if n < 800 else (5 if n < 4000 else 8)
    qs = np.linspace(0, 1, K + 1)[1:-1]
    edges = np.quantile(x, qs)
    ctx = np.searchsorted(edges, x, side="right")
    if return_edges:
        return ctx, edges, K
    return ctx


def context_residuals(x, y, ctx, K, trim_q=0.0):
    """Additive-mechanism residuals per context: r = y - context-mean of y
    (context-constant m_hat; documented estimator). Winsorizes residuals at
    trim_q/1-trim_q WITHIN context when trim_q > 0."""
    out = []
    for c in range(K):
        m = ctx == c
        r = y[m] - y[m].mean()
        if trim_q > 0 and len(r) >= 10:
            lo, hi = np.quantile(r, [trim_q, 1 - trim_q])
            r = np.clip(r, lo, hi)
        out.append(r)
    return out


def _median_bandwidth(resids):
    pool = np.concatenate(resids)
    pool = pool[np.isfinite(pool)]
    if len(pool) < 10:
        return 1.0
    sub = pool[::max(1, len(pool) // 2000)]
    d = np.abs(sub[:, None] - sub[None, :])
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return max(med * 1.06, 1e-6)


def _gram(points, bw):
    """RBF Gram matrix and cross matrices against component grid."""
    z = (points[:, None] - points[None, :]) / bw
    return np.exp(-0.5 * z ** 2)


# ------------------------------------------------------------------ K1 core

class K1Witness:
    """Kernelized forcing-divergence witness over a shared coupling grid."""

    def __init__(self, resids, G=32, bw=None, fw_iters=300, m_cap=600):
        self.K = len(resids)
        self.G = G
        self.fw_iters = fw_iters
        self.bw = bw if bw is not None else _median_bandwidth(resids)
        self.m_cap = m_cap
        # component grid: standard normal quantiles
        self.t = np.quantile(np.random.default_rng(7).normal(size=200000),
                             (np.arange(G) + 0.5) / G)
        # per-context mean embeddings against grid components and self-Gram
        self.b = []      # b_c[j] = <mu_c, nu_j>
        self.A = []      # A_c[j,j'] = <nu_j, nu_j'> (same for all c)
        self.self_mmd_const = []
        Ky = _gram(self.t, self.bw)
        self.A = Ky.copy()
        for r in resids:
            if len(r) == 0:
                r = np.zeros(1)
            rs = r[:min(len(r), self.m_cap)]
            k_cross = np.exp(-0.5 * ((rs[:, None] - self.t[None, :]) /
                                     self.bw) ** 2).mean(axis=0)
            self.b.append(k_cross)
            Krr = _gram(rs[:min(len(rs), 400)], self.bw)
            self.self_mmd_const.append(float(Krr.mean()))

    def objective(self, w, c=None):
        """sum_c MMD^2(mu_c, nu_c(w)) (constants included)."""
        idx = range(self.K) if c is None else [c]
        tot = 0.0
        for cc in idx:
            tot += float(w @ self.A @ w - 2 * self.b[cc] @ w +
                         self.self_mmd_const[cc])
        return tot

    def solve(self, w_init=None, smo_rounds=None):
        """Frank-Wolfe on the simplex for min_w sum_c MMD^2(mu_c, nu_c(w)).
        Objective (up to constants): f(w) = K * w'Aw - 2 * (sum_c b_c)'w.
        Exact line-search FW followed by pairwise (SMO) refinement."""
        s = np.sum(self.b, axis=0)
        w = np.ones(self.G) / self.G if w_init is None else \
            np.asarray(w_init, float).copy()
        w = np.maximum(w, 0)
        w /= max(w.sum(), 1e-300)
        prev = np.inf
        for k in range(self.fw_iters):
            grad = 2 * self.K * (self.A @ w) - 2 * s
            j = int(np.argmin(grad))
            d = np.zeros(self.G)
            d[j] = 1.0
            # exact step via line search on the segment (convex quadratic):
            num = float(grad @ (d - w))
            den = float(2 * self.K * (d - w) @ self.A @ (d - w))
            gamma = 0.0 if den <= 0 else max(0.0, min(1.0,
                                                      -num / den))
            if gamma == 0.0:
                break
            w = (1 - gamma) * w + gamma * d
            val = self.objective(w)
            if abs(prev - val) < 1e-14 * max(1.0, abs(val)):
                break
            prev = val
        w = self._pairwise_polish(w, max_rounds=smo_rounds or 300)
        self.w_star = np.maximum(w, 0)
        self.w_star /= max(self.w_star.sum(), 1e-300)
        self.value = max(self.objective(self.w_star), 0.0)
        return self.value

    def _pairwise_polish(self, w, max_rounds=300, tol=1e-15):
        """Exact pairwise coordinate optimization over the simplex:
        repeatedly move mass along the best pair direction e_i - e_j with
        closed-form step, guaranteeing monotone descent to the QP optimum."""
        s = np.sum(self.b, axis=0)
        K = self.K
        prev = self.objective(w)
        for _ in range(max_rounds):
            g = 2 * K * (self.A @ w - s / K)
            # d_ij = e_i - e_j ; decrease along gamma:
            # f(w + g*d) = f + gamma*(g_i - g_j) + gamma^2 * K * c_ij
            Gd = self.A  # quadratic coefficients via diag/offdiag
            gi = g[:, None] - g[None, :]
            cij = np.diag(Gd)[:, None] + np.diag(Gd)[None, :] \
                - 2 * Gd
            den = 2 * K * np.maximum(cij, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                gam = -gi / np.where(den > 0, den, 1.0)
            hi = np.tile(w, (self.G, 1))          # gamma <= w_j
            lo = -np.tile(w[:, None], (1, self.G))  # gamma >= -w_i
            gam = np.clip(np.where(den > 0, gam, 0.0),
                          np.minimum(lo, 0.0) - 1e-12,
                          np.maximum(hi, 0.0) + 1e-12)
            dec = gi * gam + 0.5 * den * gam ** 2
            np.fill_diagonal(dec, 0.0)
            i, j = np.unravel_index(np.argmin(dec), dec.shape)
            best_dec = -dec[i, j]
            if best_dec <= tol * max(1.0, abs(prev)):
                break
            gamma = gam[i, j]
            w = w.copy()
            w[i] += gamma
            w[j] -= gamma
            prev -= best_dec
        return w


def k1_witness(x, y, trim_q=0.0, G=32, K=None, return_model=False):
    """Point statistic T_K1."""
    ctx = make_contexts(x, K=K)
    Kr = len(np.unique(ctx))
    resids = context_residuals(x, y, ctx, Kr, trim_q=trim_q)
    model = K1Witness(resids, G=G)
    val = model.solve()
    if return_model:
        return val, model
    return val


def k1_multiplier_bootstrap(x, y, B=199, trim_grid=(0.0,), G_boot=16, K=None,
                            rng=None, bmap=None):
    """Wild-bootstrap draws of T_K1 (Rademacher multipliers on residuals,
    pipeline re-run incl. inner optimization). bmap: optional
    {trim_q: B_eff} per-trim draw budgets (D8). Returns (B,) array for the
    FIRST trimming level plus dict of arrays per trimming level."""
    rng = rng or np.random.default_rng(20260827)
    ctx = make_contexts(x, K=K)
    Kr = len(np.unique(ctx))
    base_resids = context_residuals(x, y, ctx, Kr, trim_q=0.0)
    base_model = K1Witness(base_resids, G=G_boot)
    base_model.solve()
    w_star = base_model.w_star
    draws = {}
    for tq in trim_grid:
        b_eff = B if bmap is None else int(bmap.get(tq, B))
        vals = np.empty(b_eff)
        for b in range(b_eff):
            pert = [(r * rng.choice([-1.0, 1.0], size=len(r))) for r in
                    base_resids]
            if tq > 0:
                pert = [_winsor(r, tq) for r in pert]
            m = K1Witness(pert, G=G_boot, bw=base_model.bw)
            vals[b] = m.solve(w_init=w_star, smo_rounds=80)
        draws[tq] = vals
    return draws[list(trim_grid)[0]], draws


def _winsor(r, q):
    lo, hi = np.quantile(r, [q, 1 - q])
    return np.clip(r, lo, hi)


# ------------------------------------------------------------------ K2 core

def _w2sq(a, b):
    """Exact 1-D empirical squared Wasserstein-2 (sort-based)."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ai = np.quantile(a, np.linspace(0, 1, n))
    bi = np.quantile(b, np.linspace(0, 1, n))
    return float(((ai - bi) ** 2).mean())


def k2_witness(x, y, trim_q=0.0, K=None, return_parts=False):
    """Transport-defect witness: max standardized excess of consecutive-
    context W2^2 over the location-gap prediction."""
    ctx = make_contexts(x, K=K)
    Kr = len(np.unique(ctx))
    resids = context_residuals(x, y, ctx, Kr, trim_q=trim_q)
    means = np.array([r.mean() for r in resids])
    vars_ = np.array([r.var() for r in resids])
    d = np.array([_w2sq(resids[c], resids[c + 1]) for c in range(Kr - 1)])
    gap = np.diff(means) ** 2
    if len(d) < 2:
        val = 0.0
    else:
        # excess over drift prediction, standardized by pooled sampling scale
        resid_excess = np.maximum(d - gap, 0.0)
        se = np.sqrt((vars_[:-1]**2 / np.array([len(r) for r in resids[:-1]])
                      + vars_[1:]**2 /
                      np.array([len(r) for r in resids[1:]])) / 2.0) * 2.0
        se = np.maximum(se, 1e-12)
        val = float(np.max(resid_excess / se))
    if return_parts:
        return val, {"d": d, "gap": gap, "means": means, "vars": vars_}
    return val


def k2_multiplier_bootstrap(x, y, B=199, trim_grid=(0.0,), K=None,
                            rng=None, bmap=None):
    """bmap: optional {trim_q: B_eff} per-trim draw budgets (D8)."""
    rng = rng or np.random.default_rng(20260828)
    ctx = make_contexts(x, K=K)
    Kr = len(np.unique(ctx))
    base = context_residuals(x, y, ctx, Kr, trim_q=0.0)
    draws = {}
    for tq in trim_grid:
        b_eff = B if bmap is None else int(bmap.get(tq, B))
        vals = np.empty(b_eff)
        for b in range(b_eff):
            pert = [(r * rng.choice([-1.0, 1.0], size=len(r))) for r in base]
            if tq > 0:
                pert = [_winsor(r, tq) for r in pert]
            vals[b] = k2_from_resids(pert)
        draws[tq] = vals
    return draws[list(trim_grid)[0]], draws


def k2_from_resids(resids):
    means = np.array([r.mean() for r in resids])
    vars_ = np.array([r.var() for r in resids])
    d = np.array([_w2sq(resids[c], resids[c + 1])
                  for c in range(len(resids) - 1)])
    gap = np.diff(means) ** 2
    if len(d) < 2:
        return 0.0
    ex = np.maximum(d - gap, 0.0)
    se = np.sqrt((vars_[:-1]**2 / np.array([len(r) for r in resids[:-1]]) +
                  vars_[1:]**2 /
                  np.array([len(r) for r in resids[1:]])) / 2.0) * 2.0
    return float(np.max(ex / np.maximum(se, 1e-12)))


# ------------------------------------------------------------- HSIC baseline

def hsic_stat(x, y, bw=None):
    """Biased HSIC with RBF kernels (baseline CIT statistic)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if bw is None:
        sx = np.median(np.abs(x - np.median(x))) or 1.0
        sy = np.median(np.abs(y - np.median(y))) or 1.0
        bw_x, bw_y = max(sx * 1.06, 1e-6), max(sy * 1.06, 1e-6)
    else:
        bw_x, bw_y = bw
    Kx = np.exp(-0.5 * ((x[:, None] - x[None, :]) / bw_x) ** 2)
    Ky = np.exp(-0.5 * ((y[:, None] - y[None, :]) / bw_y) ** 2)
    Kxc = Kx - Kx.mean(axis=0) - Kx.mean(axis=1) + Kx.mean()
    Kyc = Ky - Ky.mean(axis=0) - Ky.mean(axis=1) + Ky.mean()
    return float((Kxc * Kyc).sum() / n ** 2)


def hsic_bootstrap(x, y, B=199, rng=None):
    """Multiplier bootstrap for HSIC (Rademacher weights on feature
    products' low-rank approximation via direct reweighting)."""
    rng = rng or np.random.default_rng(20260829)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    sx = np.median(np.abs(x - np.median(x))) or 1.0
    sy = np.median(np.abs(y - np.median(y))) or 1.0
    Kx = np.exp(-0.5 * ((x[:, None] - x[None, :]) / max(sx * 1.06, 1e-6))**2)
    Ky = np.exp(-0.5 * ((y[:, None] - y[None, :]) / max(sy * 1.06, 1e-6))**2)
    Kxc = Kx - Kx.mean(axis=0) - Kx.mean(axis=1) + Kx.mean()
    Kyc = Ky - Ky.mean(axis=0) - Ky.mean(axis=1) + Ky.mean()
    prodc = Kxc * Kyc
    draws = np.empty(B)
    for b in range(B):
        xi = rng.choice([-1.0, 1.0], size=n)
        draws[b] = float(xi @ prodc @ xi) / n ** 2
    return draws
