"""Phase 4 unit tests: DGP extensions, adversarial null class membership,
driver group construction, and aggregate-side helpers."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase2_dgps as dg  # noqa: E402
from models import build_iv_A_general  # noqa: E402
from witnesses import kl_em_batch, slack_and_feasible  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
P4 = json.loads((ROOT / "configs" / "phase4.json").read_text())


# ------------------------------------------------- D-P4.1 tail extensions

def test_bursty_null_is_iv_form_and_deterministic_seedable():
    rng1 = np.random.default_rng(700000)
    rng2 = np.random.default_rng(700000)
    r1, m1 = dg.sample_rows_null(rng1, 500, 2, 2, 4, "chain", p_conc=0.3)
    r2, m2 = dg.sample_rows_null(rng2, 500, 2, 2, 4, "chain", p_conc=0.3)
    assert np.array_equal(r1, r2)
    # bursty: across fresh seeds some context share drops below the WP2.2
    # interior floor 0.10 (Dirichlet(0.3) is U-shaped on kz=2)
    mins = [min(np.array(dg.sample_rows_null(
        np.random.default_rng(700000 + s), 50, 2, 2, 4, "chain",
        p_conc=0.3)[1]["p_z"])) for s in range(20)]
    assert min(mins) < 0.10
    # rows are valid (z,x,y) alphabet values
    assert r1[:, 0].max() < 2 and r1[:, 1].max() < 2 and r1[:, 2].max() < 4


def test_default_null_unchanged_by_extension():
    """p_conc=None reproduces the interior-constrained WP 2.2 behavior."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        _, meta = dg.sample_rows_null(rng, 50, 2, 2, 3, "u_on_x")
        assert all(0.10 < p < 0.90 for p in meta["p_z"])


def test_boundary_anchor_mixture_rho0_is_feasible():
    """tail=t3 anchor with anchor_conc=0.3 remains EXACTLY feasible at
    rho=0 (product-form coupling pushforward)."""
    mix = dg.MixtureCell(2, 2, 5, alt_seed=20260826, anchor_conc=0.3)
    e0 = mix.population_conditional(0.0).reshape(-1)
    M = build_iv_A_general(2, 2, 5).astype(float)
    slack, feas = slack_and_feasible(M, e0)
    assert feas and slack <= 1e-9


def test_boundary_anchor_mixture_rho1_is_contextual():
    mix = dg.MixtureCell(2, 2, 5, alt_seed=20260826, anchor_conc=0.3)
    e1 = mix.population_conditional(1.0).reshape(-1)
    M = build_iv_A_general(2, 2, 5).astype(float)
    val, _ = kl_em_batch(M, e1[None, :])
    assert float(val[0]) > 0.01      # conflict law fires


# ------------------------------------------------- adversarial null class

def test_A1_observable_null_mixture_is_feasible():
    """Shared-p_z observable mixture of two chain laws: the per-context
    conditional mixture has equal weights, hence is the pushforward of the
    coupling mixture (convexity of the coupling set)."""
    rng = np.random.default_rng(0)
    kz, kx, ky = 2, 2, 5
    blk = kx * ky
    while True:
        p_z = rng.dirichlet(np.ones(kz))
        if np.all((p_z > 0.10) & (p_z < 0.90)):
            break
    cond = np.zeros((kz, blk))
    for _ in range(2):
        samp, pz = dg.sample_null_mechanism(rng, kz, kx, ky, "chain",
                                            p_z_given=p_z)
        for z in range(kz):
            for u in (0, 1):
                x, y = samp(z, u)
                cond[z, x * ky + y] += 0.5 * 0.5
    M = build_iv_A_general(kz, kx, ky).astype(float)
    slack, feas = slack_and_feasible(M, cond.reshape(-1))
    assert feas and slack <= 1e-9


def test_A2_det_nonlinear_chain_is_feasible():
    from run_p4_discrete import _det_nonlinear_rows
    kz, kx, ky = 2, 2, 5
    blk = kx * ky
    meta_cond = np.zeros((kz, blk))
    for z in range(kz):
        for x in range(kx):
            y = (3 * x ** 2 + 2 * z + 1) % ky
            meta_cond[z, x * ky + y] = 0.5
    M = build_iv_A_general(kz, kx, ky).astype(float)
    _, feas = slack_and_feasible(M, meta_cond.reshape(-1))
    assert feas
    rng = np.random.default_rng(1)
    rows = _det_nonlinear_rows(rng, 20000, kz, kx, ky)
    assert rows.shape == (20000, 3)
    assert rows[:, 0].max() < kz and rows[:, 1].max() < kx \
        and rows[:, 2].max() < ky


# --------------------------------------------------- group constructors

def test_discrete_group_grids_match_config():
    from run_p4_discrete import (load_p4_cfg, make_null_groups,
                                 make_alt_groups, make_adv_groups)
    p4, cfg4, cfg2 = load_p4_cfg()
    dgx = p4["discrete_grid"]
    assert len(make_null_groups(p4, cfg2, cfg4)) == \
        len(dgx["cells"]) * len(dgx["n_grid"]) * len(dgx["null_kinds"]) * \
        len(dgx["tails"])
    assert len(make_alt_groups(p4)) == \
        len(dgx["cells"]) * len(dgx["n_grid"]) * len(dgx["rho_grid"]) * \
        len(dgx["tails"])
    assert len(make_adv_groups(p4)) == \
        len(p4["adversarial_nulls"]["discrete"])


def test_continuous_group_grids_match_config():
    from run_p4_continuous import (make_null_groups, make_alt_groups,
                                   make_adv_groups)
    p4 = P4
    cg = p4["continuous_grid"]
    # D-P4.9 (v4): quadratic detrend changes every n; full grid re-run
    assert len(make_null_groups(p4)) == len(cg["n_grid"]) * \
        len(cg["cells_d"]) * len(cg["noise"]) * len(cg["null_kinds"])
    assert len(make_alt_groups(p4)) == len(cg["n_grid"]) * \
        len(cg["cells_d"]) * len(cg["noise"]) * len(cg["alt_kinds"]) * \
        len(cg["b_grid"])
    assert len(make_adv_groups(p4)) == \
        len(p4["adversarial_nulls"]["continuous"])


def test_adversarial_continuous_samples_are_finite_and_sized():
    from run_p4_continuous import _adv_sample
    for spec in P4["adversarial_nulls"]["continuous"]:
        rng = np.random.default_rng(900000)
        x, y = _adv_sample(rng, spec["n"], spec)
        assert len(x) == len(y) == spec["n"]
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))


def test_adversarial_continuous_perm_calibration_holds_size():
    """Smoke-level size check on one adversarial config (B4): at B=49 the
    permutation calibration must not fire on most null draws."""
    from run_p4_continuous import _adv_sample
    from continuous_witness import k1_k2_perm_calibration
    spec = {"id": "B4_heavy_curvature", "n": 2000, "d": 2, "noise": "t3"}
    rej = 0
    trials = 10
    for s in range(trials):
        rng = np.random.default_rng(900000 + s)
        x, y = _adv_sample(rng, spec["n"], spec)
        obs, dr = k1_k2_perm_calibration(
            x, y, B=49, trims=(0.05,),
            rng=np.random.default_rng(s + 7300000))
        rej += int(obs[0.05]["k1"] > np.quantile(dr[0.05]["k1"], 0.95))
    assert rej <= 4        # binomial(10, 0.05) upper tail guard
