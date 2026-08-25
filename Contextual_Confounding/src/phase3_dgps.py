"""Phase 3 seeded DGPs (WP 3.2 / 3.3).

Nulls (unconfounded within the declared additive-noise class):
- null_gauss    : X = a*xi + e_X ; Y = c*X + e_Y (independent noises)
- null_nonparam : Y = g(X) + e_Y with smooth seeded g

Alternatives (B2 section 2.2 family - support of (X,Y) is R^2,
contractible, persistent-homology blind by construction):
- conf_lin      : X = U + e_X ; Y = b*X + U + e_Y
- conf_nonlin   : X = U + e_X ; Y = b*X + sin(1.5 U) + 0.5 U + e_Y

Noise families for e_*: gauss, t3 (student-t df=3, standardized),
lognorm (standardized). Nuisance covariates W extend dimension to d.
All generators return (X, Y, W) arrays with W possibly zero-width.
"""
import numpy as np


def _noise(rng, n, kind):
    if kind == "gauss":
        return rng.normal(size=n)
    if kind == "t3":
        z = rng.standard_t(3, size=n)
        return (z - z.mean()) / max(z.std(), 1e-9)
    if kind == "lognorm":
        z = rng.lognormal(size=n)
        return (z - z.mean()) / max(z.std(), 1e-9)
    raise ValueError(kind)


def _nuisance(rng, n, d):
    return rng.normal(size=(n, max(d - 2, 0)))


def sample_null(rng, n, d, noise="gauss", kind="null_gauss"):
    a = rng.choice([-2.0, -1.0, 1.0, 2.0])
    c = rng.choice([-1.5, -1.0, 1.0, 1.5])
    xi = rng.normal(size=n)
    ex = _noise(rng, n, noise)
    ey = _noise(rng, n, noise)
    x = a * 0.7 * xi + ex
    if kind == "null_gauss":
        y = c * x + ey
    elif kind == "null_nonparam":
        y = 0.8 * np.tanh(x) + 0.4 * np.sin(x) + ey
    else:
        raise ValueError(kind)
    W = _nuisance(rng, n, d)
    return x, y, W


def sample_confounded(rng, n, d, b, noise="gauss", kind="conf_lin",
                      u_scale=1.0):
    u = rng.normal(size=n) * u_scale
    ex = _noise(rng, n, noise)
    ey = _noise(rng, n, noise)
    x = u + ex
    if kind == "conf_lin":
        y = b * x + u + ey
    elif kind == "conf_nonlin":
        y = b * x + np.sin(1.5 * u) + 0.5 * u + ey
    else:
        raise ValueError(kind)
    W = _nuisance(rng, n, d)
    return x, y, W


def population_contextuality_probe(kind, b=0.0, noise="gauss", n=int(4e6),
                                   seed=999, K=5):
    """Large-sample witness value approximating the population functional
    (used to classify contamination cells and check witness monotonicity
    without sampling noise)."""
    rng = np.random.default_rng(seed)
    if kind.startswith("null"):
        x, y, _ = sample_null(rng, n, 2, noise=noise, kind=kind)
    else:
        x, y, _ = sample_confounded(rng, n, 2, b, noise=noise, kind=kind)
    return x, y
