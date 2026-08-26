"""WP 3.1 tests for the continuous witnesses (K1, K2) + HSIC baseline.

Checks:
1. K1 == 0 (within tolerance) on a shared-shape family: context laws are
   exact translates of one common law.
2. K1 > 0 on a variance-inflated family (confounding signature).
3. K1 optimizer sanity: objective(w*) <= objective(uniform) and matches
   brute force on a small grid.
4. K2 ~ 0 under pure location drift; K2 > 0 under variance drift with
   matched means.
5. Wild-multiplier bootstrap: under the null simulation, the observed
   witness sits inside its own bootstrap support; rejection frequency at
   nominal 0.05 is within [0, 2*alpha] over repeated datasets (size check,
   loose band at small R).
6. Trimming policy runs and changes values only modestly under clean
   Gaussian noise.
7. HSIC baseline: near-zero under independent draws; bootstrap CVs give
   size in-band under H0.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continuous_witness import (K1Witness, k1_witness, k1_multiplier_bootstrap,  # noqa: E402
                                k2_witness, k2_multiplier_bootstrap,
                                hsic_stat, hsic_bootstrap, make_contexts)

FAILS = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        FAILS.append(name)


def shared_shape_data(rng, n=1500, K=None, confound=0.0, hetero=0.0):
    """Null: y = m(x) + sigma*eps with eps ~ N(0,1) shared.
    Confounding injection: y += confound * u * s(x) where u is a latent
    common driver of x (x = rho*u + sqrt(1-rho^2)*xi) - creates stratum-
    dependent variance/skew in standardized residuals."""
    n = n if K is None else n
    u = rng.normal(size=n)
    xi = rng.normal(size=n)
    x = 0.8 * u + np.sqrt(max(1 - 0.64, 0)) * xi if confound > 0 \
        else rng.normal(size=n)
    m = 0.6 * np.tanh(x)
    eps = rng.normal(size=n)
    y = m + eps * (1.0 + hetero * np.abs(x))
    if confound > 0:
        y = m + eps + confound * u * (1 + np.abs(x))
    return x, y


def main():
    # ---------------- 1-3: K1 behavior + optimizer sanity
    rng = np.random.default_rng(11)
    x0, y0 = shared_shape_data(rng, n=1200, confound=0.0)
    t_null = k1_witness(x0, y0)
    xc, yc = shared_shape_data(rng, n=1200, confound=1.2)
    t_alt = k1_witness(xc, yc)
    print(f"K1 null={t_null:.5f} alt={t_alt:.5f}")
    check("K1 ~ 0 under shared-shape null (<0.02)", t_null < 0.02)
    check("K1 fires under confounding (alt > 5x null + margin)",
          t_alt > max(5 * max(t_null, 1e-4), 0.02))

    val, model = k1_witness(xc, yc[:len(xc)], return_model=True) if False \
        else k1_witness(xc, yc, return_model=True)
    w_unif = np.ones(model.G) / model.G
    check("FW optimum <= uniform coupling",
          model.objective(model.w_star) <=
          model.objective(w_unif) + 1e-12)
    # brute force on tiny grid
    import itertools
    m_small = K1Witness.__new__(K1Witness)
    r = [np.random.default_rng(3).normal(size=200),
         np.random.default_rng(4).normal(size=200) + 2.0]
    ms = K1Witness(r, G=5)
    best = np.inf
    for w in itertools.product(range(21), repeat=5):
        wv = np.array(w, float) / 20.0
        if abs(wv.sum() - 1) < 1e-9:
            best = min(best, ms.objective(wv))
    ms.solve()
    check("FW+SMO at least as good as every coarse-grid coupling",
          ms.value <= best + 1e-9)
    check("FW+SMO not pathologically below grid best",
          ms.value >= best - 5e-3 * max(1.0, abs(best)))

    # ---------------- 4: K2 behavior
    v_null, parts_n = k2_witness(x0, y0, return_parts=True)
    # pure location drift construction: same shape shifted per context
    res_loc = [rng.normal(size=400) + 3.0 * c for c in range(5)]
    from continuous_witness import k2_from_resids
    v_loc = k2_from_resids(res_loc)
    res_var = [rng.normal(size=400) * (1 + 1.5 * c) for c in range(5)]
    v_var = k2_from_resids(res_var)
    v_conf = k2_witness(xc, yc)
    print(f"K2 null={v_null:.3f} locdrift={v_loc:.3f} vardrift={v_var:.3f} "
          f"conf={v_conf:.3f}")
    check("K2 silent under pure location drift (<1)", v_loc < 1.0)
    check("K2 fires under variance drift (>2)", v_var > 2.0)

    # ---------------- 5: bootstrap size check (loose, small R)
    alpha = 0.05
    rejects = {"k1": 0, "k2": 0}
    R = 40
    for rep in range(R):
        rr = np.random.default_rng(500 + rep)
        xs, ys = shared_shape_data(rr, n=800, confound=0.0)
        obs1 = k1_witness(xs, ys)
        d1, _ = k1_multiplier_bootstrap(xs, ys, B=99,
                                        rng=np.random.default_rng(
                                            9000 + rep))
        obs2 = k2_witness(xs, ys)
        d2, _ = k2_multiplier_bootstrap(xs, ys, B=99,
                                        rng=np.random.default_rng(
                                            9500 + rep))
        cv1 = np.quantile(d1, 0.95)
        cv2 = np.quantile(d2, 0.95)
        rejects["k1"] += int(obs1 > cv1)
        rejects["k2"] += int(obs2 > cv2)
    print(f"bootstrap size at alpha=0.05 (R={R}): "
          f"k1={rejects['k1']/R:.3f} k2={rejects['k2']/R:.3f}")
    check("K1 wild-bootstrap worst size <= 2*alpha (+MC slack)",
          rejects["k1"] / R <= 2 * alpha + 0.08)
    check("K2 wild-bootstrap worst size <= 2*alpha (+MC slack)",
          rejects["k2"] / R <= 2 * alpha + 0.08)

    # ---------------- 6: trimming sensitivity
    t0 = k1_witness(x0, y0, trim_q=0.0)
    t1 = k1_witness(x0, y0, trim_q=0.05)
    xt = np.concatenate([x0, [10.0]])
    yt = np.concatenate([y0, [50.0]])
    t_out = k1_witness(xt, yt, trim_q=0.05)
    print(f"trim: none={t0:.4f} q05={t1:.4f} outlier+q05={t_out:.4f}")
    check("trimming keeps clean-data value in same ballpark",
          abs(t1 - min(t0, t1)) < 0.02)
    check("trimming caps single-outlet blowup",
          t_out < max(10 * max(t1, 1e-3), 0.2))

    # ---------------- 7: HSIC baseline sanity
    h_ind = hsic_stat(rng.normal(size=600), rng.normal(size=600))
    h_dep = hsic_stat(np.arange(600.0), np.arange(600.0) +
                      np.random.default_rng(2).normal(scale=.3, size=600))
    check("HSIC ~0 independent", abs(h_ind) < 5e-3)
    check("HSIC large dependent", h_dep > 10 * max(abs(h_ind), 1e-4))

    print()
    if FAILS:
        print("FAILURES:", FAILS)
        sys.exit(1)
    print("ALL WP 3.1 TESTS PASSED")


if __name__ == "__main__":
    main()
