"""CCX core models: response-variable maps, exact facet machinery.

Conventions (see memos/formalization_memo.md):
- Binary IV scenario S_IV: contexts z in {0,1}; observable conditional vector
  e (8,) ordered (z, x, y) with z-major, each z-block summing to 1.
- Response tables: r_X indexed ix = 2*r_X(0) + r_X(1); r_Y indexed iy likewise;
  q index = 4*ix + iy (16 coords).
- Atlas structures use the generic joint form: observable P(o) over all observed
  configs; q ranges over (exogenous coords incl. latents, response tables).
"""
import json
import re
from itertools import product
from pathlib import Path

import numpy as np
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load_seeds():
    with open(CONFIGS / "seeds.json") as f:
        return json.load(f)


# ---------------------------------------------------------------- S_IV (16-dim)

def iv_response_tables():
    """Return (tables_ix, tables_iy) mapping index -> (value at 0, value at 1)."""
    tabs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return tabs, tabs


def build_iv_A():
    """A: (8, 16) int matrix; row index = 4*z + 2*x + y."""
    tabs_x, tabs_y = iv_response_tables()
    A = np.zeros((8, 16), dtype=np.int64)
    for ix, tx in enumerate(tabs_x):
        for iy, ty in enumerate(tabs_y):
            ry = {0: ty[0], 1: ty[1]}
            for z in (0, 1):
                x = tx[z]
                y = ry[x]
                A[4 * z + 2 * x + y, 4 * ix + iy] = 1
    return A


def iv_vertices():
    """Images of the 16 deterministic response tables (int, shape (16, 8))."""
    A = build_iv_A()
    V = np.zeros((16, 16), dtype=np.int64)
    np.fill_diagonal(V, 1)
    return (A @ V.T).T


# ------------------------------------------------------- exact facet machinery

def _nullspace_exact(rows):
    """Exact rational nullspace basis (list of tuples) of a list of Fraction rows."""
    m = len(rows)
    n = len(rows[0])
    mat = [list(r) for r in rows]
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if mat[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        mat[r] = [x / pv for x in mat[r]]
        for i in range(m):
            if i != r and mat[i][c] != 0:
                f = mat[i][c]
                mat[i] = [a - f * b for a, b in zip(mat[i], mat[r])]
        pivots.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for fc in free:
        vec = [Fraction(0)] * n
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            vec[pc] = -mat[i][fc]
        basis.append(vec)
    return basis


def _canonical(n_int):
    g = 0
    for x in n_int:
        g = np.gcd(g, abs(int(x)))
    if g > 0:
        n_int = n_int // g
    for x in n_int:
        if x != 0:
            if x < 0:
                n_int = -n_int
            break
    return tuple(int(x) for x in n_int)


def affine_free_columns(M):
    """Exact affine constraints satisfied by every point of P = {Mq: q in
    simplex}: rows v with v^T M = lambda 1^T, computed as the exact rational
    nullspace of [M^T | -1]. Returns (free_cols, B): projecting onto
    free_cols is injective on P."""
    D, Q = M.shape
    N = []
    for qi in range(Q):
        N.append([Fraction(int(x)) for x in M[:, qi]] + [Fraction(-1)])
    ns = _nullspace_exact(N)
    B = np.array([[int(x[i]) for i in range(D)] for x in ns], dtype=np.int64)
    pivots = []
    r = 0
    mat = [[Fraction(int(x)) for x in row] for row in B]
    for c in range(D):
        piv = None
        for i in range(r, len(mat)):
            if mat[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        mat[r] = [x / pv for x in mat[r]]
        for i in range(len(mat)):
            if i != r and mat[i][c] != 0:
                f = mat[i][c]
                mat[i] = [a - f * b for a, b in zip(mat[i], mat[r])]
        pivots.append(c)
        r += 1
        if r == len(mat):
            break
    free = [c for c in range(D) if c not in pivots]
    return free, B


def discover_facets(M, seed=0, active_tol=1e-7):
    """Facets of P = conv{columns of M} (columns are integer vertices).
    Pipeline: (1) exact affine reduction to free coordinates (injective on P);
    (2) qhull candidates (complete for the hull; triangulation pieces of a
    non-simplicial facet share one hyperplane); (3) exact rational snapping
    through support vertices + exact sign validation. Returns (H, b) int arrays,
    H p <= b valid on P (coefficients zero-padded at pivot coordinates)."""
    from scipy.spatial import ConvexHull
    from fractions import Fraction

    D, Q = M.shape
    verts = M.T.astype(np.int64)  # (Q, D)
    free, _B = affine_free_columns(M)
    Vaf = verts[:, free]

    hull = ConvexHull(Vaf.astype(float))
    facets = {}
    for simp in hull.simplices:
        # each simplex row lists vertices of one triangulation piece; all
        # pieces of a non-simplicial facet share its exact hyperplane
        base = Vaf[simp[0]]
        rows = [[Fraction(int(x)) for x in Vaf[si] - base] for si in simp[1:]]
        ns = _nullspace_exact(rows)
        if len(ns) != 1:
            continue  # degenerate piece
        den = 1
        for x in ns[0]:
            den = den * x.denominator // int(np.gcd(den, x.denominator))
        n_int = np.array([int(x * den) for x in ns[0]], dtype=np.int64)
        off_int = int(n_int @ Vaf[simp[0]])
        key = _canonical(n_int)
        if key in facets:
            continue
        valsi = Vaf @ n_int
        if valsi.max() > off_int:
            n_int, off_int = -n_int, -off_int
            key = _canonical(n_int)
            valsi = Vaf @ n_int
        assert valsi.max() <= off_int, "orientation/validation failed"
        facets[key] = (n_int, off_int)
    H = np.zeros((len(facets), D), dtype=np.int64)
    b = np.zeros(len(facets), dtype=np.int64)
    for i, (n_int, off_int) in enumerate(facets.values()):
        H[i, free] = n_int
        b[i] = off_int
    return H, b


def classify_trivial(H, b, M, n_samples=200, seed=1):
    """A facet is 'trivial' if its inequality is affinely equivalent to
    -p_o <= 0 for some observed coordinate o on the polytope (positivity facet
    of the ambient simplex rather than a structural constraint). Tested up to
    additive constants on a sampled feasible cloud."""
    rng = np.random.default_rng(seed)
    Q = M.shape[1]
    W = rng.dirichlet(np.ones(Q), size=n_samples)
    P = (M.astype(float) @ W.T).T
    out = []
    for h, off in zip(H, b):
        vals = P @ h.astype(float)
        trivial = False
        for o in range(P.shape[1]):
            diff = vals - (-P[:, o])
            if float(diff.max() - diff.min()) < 1e-9 and \
                    abs(off - float(np.median(vals))) < 1e-6:
                trivial = True
                break
        out.append(trivial)
    return np.array(out, dtype=bool)


def facets_feasible(H, b, e, tol=1e-9):
    return bool(np.all(H.astype(float) @ e <= b.astype(float) + tol))


# ------------------------------------------------------------ generic atlas

def build_structure(spec):
    """spec: dict with 'nodes': ordered dict name->dict(parents=[...], latent=bool,
    exogenous=bool (no parents handled by parents==[]), 'observed': [names].
    Endogenous nodes have mechanisms r: pa values -> {0,1}.
    Returns M (D x Q) int mapping q-simplex to joint observable P(observed),
    plus metadata about coordinate blocks."""
    nodes = spec["nodes"]
    observed = spec["observed"]
    names = list(nodes.keys())

    def table_count(nd):
        k = len(nd["parents"])
        return nd.get("card", 2) ** k

    blocks = []  # (node, kind, size)
    for nm in names:
        nd = nodes[nm]
        if nd["parents"]:
            blocks.append((nm, "resp", table_count(nd)))
        else:
            blocks.append((nm, "val", nd.get("card", 2)))
    Q = int(np.prod([b[2] for b in blocks]))
    D = 2 ** len(observed)

    obs_index = {cfg: i for i, cfg in enumerate(product([0, 1], repeat=len(observed)))}

    def decode(idx):
        out = {}
        for (nm, kind, sz) in reversed(blocks):
            out[nm] = idx % sz
            idx //= sz
        return out

    def decode_obs(i):
        cfg = next(k for k, v in obs_index.items() if v == i)
        return dict(zip(observed, cfg))

    M = np.zeros((D, Q), dtype=np.int64)
    for qi in range(Q):
        assign = decode(qi)
        # resolve values of all nodes given assignment
        resolved = {}
        for nm in names:
            nd = nodes[nm]
            if nd["parents"]:
                pa_key = tuple(resolved[p] for p in nd["parents"])
                resolved[nm] = table_entry(pa_key, assign[nm], len(nd["parents"]))
            else:
                resolved[nm] = assign[nm]
        for oi in range(D):
            ov = decode_obs(oi)
            ok = all(resolved[nm] == ov[nm] for nm in observed)
            if ok:
                M[oi, qi] = 1
    meta = {"blocks": blocks, "Q": Q, "D": D}
    return M, meta


def table_entry(pa_key, tab_idx, k):
    """Value assigned by response table tab_idx (bit position = parent config
    in mixed radix, first parent most significant) to pa_key."""
    pos = 0
    for bit in pa_key:
        pos = pos * 2 + bit
    return (int(tab_idx) >> pos) & 1


ATLAS_SPECS = {
    # A1_iv is the S_IV scenario itself (Phase 1 main run supplies its numbers);
    # listed here with method='reference'.
    "A1_iv": {
        "desc": "IV: Z->X, X->Y, U->X, U->Y; Z exogenous observed",
        "method": "reference",
        "nodes": {
            "Z": {"parents": [], "latent": False},
            "U": {"parents": [], "latent": True},
            "X": {"parents": ["Z", "U"], "latent": False},
            "Y": {"parents": ["X", "U"], "latent": False},
        },
        "observed": ["Z", "X", "Y"],
    },
    "A2_iv_direct": {
        "desc": "IV with direct Z->Y edge (exclusion broken); conditional form",
        "method": "conditional",
        "nodes": {
            "Z": {"parents": [], "latent": False},
            "U": {"parents": [], "latent": True},
            "X": {"parents": ["Z", "U"], "latent": False},
            "Y": {"parents": ["X", "Z", "U"], "latent": False},
        },
        "observed": ["Z", "X", "Y"],
    },
    "A3_proxy_collider": {
        "desc": "Pure collider Z->X<-U->Y; feasible set is a finite union of "
                "polytopes (nonconvex): MILP feasibility, reduced density",
        "method": "milp",
        "nodes": {
            "Z": {"parents": [], "latent": False},
            "U": {"parents": [], "latent": True},
            "X": {"parents": ["Z", "U"], "latent": False},
            "Y": {"parents": ["U"], "latent": False},
        },
        "observed": ["Z", "X", "Y"],
    },
    "A4_latent_mediator": {
        "desc": "Mediated latent confounding: U->X, U->Y (direct), X->M->Y; "
                "joint form, u-explicit exact parametrization",
        "method": "joint",
        "nodes": {
            "U": {"parents": [], "latent": True},
            "X": {"parents": ["U"], "latent": False},
            "M": {"parents": ["X"], "latent": False},
            "Y": {"parents": ["M", "U"], "latent": False},
        },
        "observed": ["X", "M", "Y"],
    },
}


def build_u_explicit_map(spec):
    """Exact single-latent parametrization: coordinates are
    (u, response tables of u-dependent mechanisms at that u, shared tables of
    u-independent mechanisms). Observable: joint law of observed nodes.
    Returns M (D x Q) int."""
    nodes = spec["nodes"]
    observed = spec["observed"]
    names = list(nodes.keys())
    latent_roots = [nm for nm in names
                    if nodes[nm]["latent"] and not nodes[nm]["parents"]]
    assert len(latent_roots) == 1, "single latent root required"
    U = latent_roots[0]

    def n_tables(nd):
        obs_pa = [p for p in nd["parents"] if p != U]
        k = len(obs_pa)
        return 2 ** (2 ** k)

    groups = []  # (node, 'u' or 'shared', n_tables)
    for nm in names:
        nd = nodes[nm]
        if not nd["parents"]:
            continue
        if U in nd["parents"]:
            groups.append((nm, "u", n_tables(nd)))
        else:
            groups.append((nm, "shared", n_tables(nd)))
    Q = 2 * int(np.prod([g[2] for g in groups])) if groups else 2

    obs_cfgs = list(product([0, 1], repeat=len(observed)))
    obs_index = {cfg: i for i, cfg in enumerate(obs_cfgs)}
    M = np.zeros((len(obs_cfgs), Q), dtype=np.int64)
    gsz = [g[2] for g in groups]

    def resolve(col):
        # decode mixed radix over (u, groups...)
        vals = {}
        rem = col
        for nm, kind, sz in reversed(groups):
            vals[nm] = rem % sz
            rem //= sz
        u = rem % 2
        resolved = {U: u}
        for nm, kind, sz in groups:
            nd = nodes[nm]
            obs_pa = [p for p in nd["parents"] if p != U]
            if kind == "u":
                t = vals[nm]
                pk = tuple(resolved[p] for p in obs_pa)
                pos = 0
                for b in pk:
                    pos = pos * 2 + b
                resolved[nm] = (int(t) >> pos) & 1
            else:
                t = vals[nm]
                pk = tuple(resolved[p] for p in nd["parents"])
                pos = 0
                for b in pk:
                    pos = pos * 2 + b
                resolved[nm] = (int(t) >> pos) & 1
        return resolved

    for qi in range(Q):
        res = resolve(qi)
        for oi, cfg in enumerate(obs_cfgs):
            ov = dict(zip(observed, cfg))
            if all(res[nm] == ov[nm] for nm in observed):
                M[oi, qi] = 1
    return M


def build_conditional_map_A2():
    """Conditional-form map for A2_iv_direct: contexts z in {0,1}; response
    coords: r_X (4 tables over z) x r_Y (8 tables over (x,z)); observable
    blocks P(x,y|z). Returns A (8 x 32) int."""
    tabs_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tabs_y = [(a, b, c, d) for a in (0, 1) for b in (0, 1)
              for c in (0, 1) for d in (0, 1)]  # y at (x,z): idx = x*2+z
    Q = len(tabs_x) * len(tabs_y)              # 4 * 16 = 64
    A = np.zeros((8, Q), dtype=np.int64)
    for ix, tx in enumerate(tabs_x):
        for iy, ty in enumerate(tabs_y):
            col = len(tabs_y) * ix + iy        # stride = |tables_y| = 16
            for z in (0, 1):
                x = tx[z]
                y = ty[x * 2 + z]
                A[4 * z + 2 * x + y, col] = 1
    return A

def build_iv_A_general(kz, kx, ky):
    """General-alphabet IV response map. Contexts z in 0..kz-1; observables
    (x, y) with x in 0..kx-1, y in 0..ky-1. Response coordinates: r_X tables
    in [kx]^{kz} (kx**kz of them) and r_Y tables in [ky]^{[kx]} (ky**kx).
    Column index = len_y * ix + iy with ix = mixed-radix code of r_X over z
    (first z most significant) and iy likewise for r_Y over x.
    Row index = kx * ky * z + kx... row = (z, x, y): idx = (ky * x + y) + kz-block:
    row = kz_block_stride*z + ky*x + y where block stride = kx*ky."""
    tabs_x = _mixed_radix_tables(kx, kz)
    tabs_y = _mixed_radix_tables(ky, kx)
    K, Q = kz * kx * ky, len(tabs_x) * len(tabs_y)
    A = np.zeros((K, Q), dtype=np.int64)
    for ix, tx in enumerate(tabs_x):
        for iy, ty in enumerate(tabs_y):
            col = len(tabs_y) * ix + iy
            for z in range(kz):
                x = tx[z]
                y = ty[x]
                A[(kx * ky) * z + ky * x + y, col] = 1
    return A


def _mixed_radix_tables(k_out, k_in):
    """All functions [k_in] -> [k_out] as tuples; tuple[t] = value at input t;
    index = mixed radix with first input most significant."""
    return [tuple((t // (k_out ** p)) % k_out for p in reversed(range(k_in)))
            for t in range(k_out ** k_in)]


def iv_q_index(kx, ky, tab_x, tab_y, n_tabs_y):
    """Column index of response-table pair (tab_x over z, tab_y over x)."""
    ix = 0
    for v in tab_x:
        ix = ix * kx + v
    iy = 0
    for v in tab_y:
        iy = iy * ky + v
    return n_tabs_y * ix + iy


# Theory flags: does the structure class possess nontrivial nested-Markov
# equality constraints on the observed law?
NM_EQUALITY_FLAGS = {
    "A1_iv": (False, "plain IV chain ADMG imposes no equality constraints"),
    "A2_iv_direct": (False, "observed ADMG is a full DAG (chain + direct edge); no equality constraints"),
    "A3_proxy_collider": (True, "implied equality: Z indep Y (collider X blocks all paths from Z to Y)"),
    "A4_latent_mediator": (False, "bidirected-style latent backdoor X<->Y with mediator chain X->M->Y: no d-separation equalities known"),
}


def collider_feasible_milp(e_joint):
    """Exact feasibility for the pure collider model Z->X<-U->Y. Latent
    cardinality unrestricted: WLOG cells are the 8 types (f_0, f_1, g) in
    {0,1}^3 (any finite latent collapses onto them). Observable blocks satisfy
      P(x,y|z) = sum_u m_u * 1[f_z(u)=x] * 1[g(u)=y].
    MILP: continuous m (8-simplex), w (2*8*2*2 >= 0); binaries
    delta[z,u,x] and gamma[u,y].
      sum_u m_u = 1;  sum_x delta[z,u,x] = 1;  sum_y gamma[u,y] = 1
      sum_{x,y} w[z,u,x,y] = m_u                       (all z,u)
      sum_u w[z,u,x,y]   = e(x,y|z)                    (all z,x,y)
      w[z,u,x,y] <= delta[z,u,x];  w[z,u,x,y] <= gamma[u,y]"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    nu = 8
    nq, nd, ng, nw = nu, 2 * nu * 2, 2 * nu, 2 * nu * 2 * 2
    off_d, off_g, off_w = nq, nq + nd, nq + nd + ng
    NV = off_w + nw
    c = np.zeros(NV)
    integrality = np.concatenate([np.zeros(nq), np.ones(nd), np.ones(ng),
                                  np.zeros(nw)])
    lb = np.zeros(NV)
    ub = np.concatenate([np.ones(nq), np.ones(nd), np.ones(ng),
                         np.full(nw, np.inf)])
    cons = []

    def con(spec, lo=None, hi=None):
        r = np.zeros(NV)
        for idx, coef in spec:
            r[idx] += coef
        cons.append(LinearConstraint(r[None, :],
                                     -np.inf if lo is None else lo,
                                     np.inf if hi is None else hi))

    dix = lambda z, u, x: off_d + (2 * nu) * z + 2 * u + x
    gix = lambda u, y: off_g + 2 * u + y
    wix = lambda z, u, x, y: off_w + (nu * 4) * z + 4 * u + 2 * x + y

    con([(i, 1.0) for i in range(nq)], lo=1.0, hi=1.0)               # sum m = 1
    for z in range(2):
        for u in range(nu):
            con([(dix(z, u, x), 1.0) for x in range(2)], lo=1.0, hi=1.0)
            con([(wix(z, u, x, y), 1.0) for x in range(2) for y in range(2)]
                + [(u, -1.0)], lo=0.0, hi=0.0)                       # = m_u
            for x in range(2):
                for y in range(2):
                    con([(wix(z, u, x, y), 1.0), (dix(z, u, x), -1.0)],
                        hi=0.0)
                    con([(wix(z, u, x, y), 1.0), (gix(u, y), -1.0)],
                        hi=0.0)
    for u in range(nu):
        con([(gix(u, y), 1.0) for y in range(2)], lo=1.0, hi=1.0)    # det g
    E = np.asarray(e_joint, dtype=float).reshape(2, 2, 2)
    p_hat = E.sum(axis=(1, 2))                      # context masses
    if np.any(p_hat <= 1e-12):
        return False
    for z in range(2):
        for x in range(2):
            for y in range(2):
                # match the context-normalized block (joint factor p_z divided
                # out; independence Z⊥U is encoded by sharing m_u across z)
                con([(wix(z, u, x, y), 1.0) for u in range(nu)],
                    lo=float(E[z, x, y] / p_hat[z]),
                    hi=float(E[z, x, y] / p_hat[z]))             # = e(.|z)
    res = milp(c=c, constraints=cons, integrality=integrality,
               bounds=Bounds(lb, ub),
               options={"time_limit": 10, "mip_rel_gap": 1e-9})
    return bool(res.success)
