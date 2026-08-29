"""Phase 4 Block E: real-data smoke (read-only feasibility demonstration).

Analyses (configs/phase4.json; D-P4.6):
- R1 card       : Z=nearc4, X=educ, Y=lwage (IV smoke, n=3010)
- R2 mroz       : Z=motheduc, X=educ, Y=lwage, in-lf subsample (n=428)
- R3 jtrain 1988: randomized grant -> train -> scrap (RCT negative
                  control; expect NO rejection)
- R4 card placebo: instrument rows permuted (relevance destroyed; expect
                  NO rejection; smoke-level FPR quantification)

Methods: discrete witnesses (kl_plugin/kl_split/cf1_margin/slack_plugin)
on quartile-binned outcomes, judged against Phase 4 envelope CVs at the
matched cell/nearest-n (conservative); continuous witnesses k1/k2 with
label-permutation calibration (B=199, trim 0.05) and the hsic baseline.

Output: results/raw/phase4/p4realdata.csv
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from continuous_witness import k1_k2_perm_calibration, hsic_resid_stat  # noqa: E402
from calibration import critical_values, fit_null_q, draw_bootstrap_counts  # noqa: E402
from witness_estimators import counts_to_conditionals  # noqa: E402
from models import build_iv_A_general  # noqa: E402
from run_wp22_calibration import observed_stats, get_gm_battery  # noqa: E402
import phase2_dgps as dg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase4"
ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]
B_PERM = 199
B_BOOT = 199
TRIM = 0.05


def load_dataset(name):
    try:
        import wooldridge as woo
    except ImportError as e:
        raise SystemExit("pip install wooldridge first (pinned in "
                         "requirements phase4 note)") from e
    return woo.data(name)


def choose_K(x, Kmax):
    """Largest K <= Kmax whose quantile strata are all nonempty (real
    covariates have heavy integer ties; duplicate quantile edges would
    create empty contexts)."""
    x = np.asarray(x, float)
    for K in range(Kmax, 1, -1):
        qs = np.linspace(0, 1, K + 1)[1:-1]
        edges = np.quantile(x, qs)
        ctx = np.searchsorted(edges, x, side="right")
        if np.all(np.bincount(ctx, minlength=K) >= 10):
            return K
    return 2


def continuous_rows(analysis_id, x, y, Kmax=5):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    K = choose_K(x, Kmax)
    out = []
    obs, dr = k1_k2_perm_calibration(
        x, y, B=B_PERM, trims=(TRIM,), K=K,
        rng=np.random.default_rng(20260901))
    for meth in ("k1", "k2"):
        cvs = critical_values(dr[TRIM][meth], ALPHA_GRID)
        row = {"analysis": analysis_id, "method": meth, "trim": TRIM,
               "B": len(dr[TRIM][meth]), "stat_obs": obs[TRIM][meth],
               "K_strata": K}
        for a in ALPHA_GRID:
            row[f"cv_{a:.2f}"] = cvs[a]
        out.append(row)
    hs = abs(hsic_resid_stat(x, y))
    row = {"analysis": analysis_id, "method": "hsic", "trim": 0.0,
           "B": 0, "stat_obs": hs, "K_strata": K}
    out.append(row)
    return out


def discrete_rows(analysis_id, z, x, y, kz, kx, ky, cell, n_env):
    """Quartile-bin y (and x if needed), evaluate discrete witnesses with
    OWN parametric-bootstrap CVs (fit null q on the observed conditionals,
    B=199 draws; D-P4.7 for the (2,2,2) RCT cell which has no Phase 4 null
    arm) plus the envelope-CV columns for the aggregator."""
    def bin_q(v, k):
        qs = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
        return np.searchsorted(qs, v, side="right")

    zb = np.asarray(z).astype(int)
    xb = bin_q(np.asarray(x, float), kx)
    yb = bin_q(np.asarray(y, float), ky)
    rows3 = np.stack([zb, xb, yb], axis=1)
    M = build_iv_A_general(kz, kx, ky)
    Mint = M.astype(int)
    Wg, cg = get_gm_battery(M, tuple(cell))
    counts = dg.counts_from_rows(rows3, kz, kx, ky)
    obs = observed_stats(M, Mint, counts, kz, kx, ky, None, None, Wg, cg)

    q_hat, push = fit_null_q(M, counts_to_conditionals(counts[None, :],
                                                       kz)[0])
    brng = np.random.default_rng(20260903)
    boot_counts = draw_bootstrap_counts(push.reshape(kz, kx * ky), counts,
                                        kz, brng, B_BOOT, "para_boot",
                                        0.7)
    draws = {s: [] for s in obs}
    for bc in boot_counts:
        for s, v in observed_stats(M, Mint, bc, kz, kx, ky, None, None,
                                   Wg, cg).items():
            draws[s].append(v)
    out = []
    for s, sv in obs.items():
        cvs = critical_values(np.array(draws[s]), ALPHA_GRID)
        row = {"analysis": analysis_id, "method": s, "trim": -1.0,
               "B": B_BOOT, "stat_obs": float(sv),
               "cell": "-".join(map(str, cell)), "n_env": n_env}
        for a in ALPHA_GRID:
            row[f"cv_{a:.2f}"] = cvs[a]
        out.append(row)
    return out


def main():
    p4 = json.loads((ROOT / "configs" / "phase4.json").read_text())
    card = load_dataset("card")
    mroz = load_dataset("mroz")
    jt = load_dataset("jtrain")
    jt88 = jt[jt["year"] == 1988].copy()

    out = []
    z = card["nearc4"].to_numpy()
    x = card["educ"].to_numpy(float)
    y = card["lwage"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    out += discrete_rows("R1_card", z[ok].astype(int), x[ok], y[ok],
                         2, 2, 4, [2, 2, 4], 2000)
    out += continuous_rows("R1_card", x[ok] - x[ok].mean(), y[ok])

    zp = card["nearc4"].to_numpy().copy()
    np.random.default_rng(20260902).shuffle(zp)
    out += discrete_rows("R4_card_placebo", zp[ok].astype(int), x[ok],
                         y[ok], 2, 2, 4, [2, 2, 4], 2000)
    # continuous placebo: the witness never sees Z, so the placebo breaks
    # the (x, y) ANM itself by permuting y (marginals preserved, shared
    # shape restored); expect NO rejection
    yp = y[ok].copy()
    np.random.default_rng(20260904).shuffle(yp)
    out += continuous_rows("R4_card_placebo", x[ok] - x[ok].mean(), yp)

    m = mroz[mroz["inlf"] == 1]
    xm = m["educ"].to_numpy(float)
    ym = m["lwage"].to_numpy(float)
    zm = (m["motheduc"] > 0).to_numpy(int)
    okm = np.isfinite(xm) & np.isfinite(ym)
    out += discrete_rows("R2_mroz", zm[okm], xm[okm], ym[okm],
                         2, 2, 4, [2, 2, 4], 2000)
    out += continuous_rows("R2_mroz", xm[okm] - xm[okm].mean(), ym[okm])

    j88 = jt[jt["year"] == 1988].dropna(subset=["grant", "scrap",
                                                "hrsemp"])
    zj = j88["grant"].to_numpy(int)                # randomized assignment
    xj = (j88["hrsemp"].to_numpy(float) > 0).astype(int)   # received training
    yj = j88["scrap"].to_numpy(float)
    yj_bin = (yj > np.median(yj)).astype(int)      # dichotomize scrap
    out += discrete_rows("R3_jtrain_rct", zj, xj.astype(float), yj_bin,
                         2, 2, 2, [2, 2, 2], 2000)

    df = pd.DataFrame(out)
    RAW.mkdir(parents=True, exist_ok=True)
    fn = RAW / "p4realdata.csv"
    df.to_csv(fn, index=False)
    print(f"wrote {len(df)} rows -> {fn}")
    print(df[["analysis", "method", "stat_obs"]].to_string(index=False))


if __name__ == "__main__":
    main()
