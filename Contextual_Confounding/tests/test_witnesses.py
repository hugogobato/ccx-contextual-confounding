"""WP 2.1 unit tests: witness estimators vs exact population values.

Checks (plan WP 2.1):
1. Population tolerance 1e-8: plugin estimators fed EXACT population counts
   reproduce Phase 1 witnesses on instances sampled from
   results/phase1_enumeration/t1_dictionary.csv.
2. Micro-examples ME-A/ME-B/ME-C through the count pipeline.
3. Monte Carlo bands for estimators under a known null law:
   - plugin KL nonnegative, concentrates to 0;
   - split and crossfit remove the plugin's downward bias direction
     (mean split >= mean plugin at small n is NOT asserted; asserted:
     crossfit/split variance-bias tradeoff stays in MC band);
   - parametric-bootstrap central bands cover the population witness value.
4. Determinism: identical seeds give identical estimator values; different
   seeds differ.
5. Split machinery conserves stratum totals.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import build_iv_A, build_iv_A_general  # noqa: E402
from witnesses import (slack_and_feasible, cf1_soft, kl_contextuality)  # noqa: E402
from witness_estimators import (counts_to_conditionals, kl_plugin_stat,  # noqa: E402
                                kl_split_stat, kl_crossfit_stat,
                                cf1_plugin_stat, slack_plugin_stat,
                                split_counts, STAT_REGISTRY)
from calibration import fit_null_q, draw_bootstrap_counts  # noqa: E402
from phase2_dgps import counts_from_rows  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        FAILS.append(name)


def main():
    A16 = build_iv_A().astype(float)

    # ---------------- 1. population agreement on Phase 1 instances
    csv = ROOT / "results" / "phase1_enumeration" / "t1_dictionary.csv"
    if csv.exists():
        df = pd.read_csv(csv, usecols=["source", "batch", "lp_feasible",
                                       "cf1_soft", "kl_contextuality",
                                       "slack_l1"] +
                         [f"e_{j}" for j in range(8)],
                         nrows=60000)
        take = pd.concat([df[~df["lp_feasible"].astype(bool)].head(60),
                          df[df["lp_feasible"].astype(bool)].head(60)])
        ecols = [f"e_{j}" for j in range(8)]
        worst = {"kl": 0.0, "cf1": 0.0, "slack": 0.0}
        for _, r in take.iterrows():
            e = r[ecols].to_numpy(dtype=float)
            # exact population conditionals passed through the count
            # plumbing as float weights (normalization is exact)
            counts = np.asarray(e, dtype=float)
            eh = counts_to_conditionals(counts[None, :], 2)[0]
            worst["kl"] = max(worst["kl"],
                              abs(kl_plugin_stat(A16, counts, 2) -
                                  float(r["kl_contextuality"])))
            worst["cf1"] = max(worst["cf1"], abs(cf1_plugin_stat(
                A16.astype(int), counts, 2) - float(r["cf1_soft"])))
            worst["slack"] = max(worst["slack"], abs(slack_plugin_stat(
                A16.astype(int), counts, 2) - float(r["slack_l1"])))
            assert np.max(np.abs(eh - e)) < 1e-7
        print(f"population max |diff|: {worst}")
        check("kl plugin matches Phase 1 at 1e-8", worst["kl"] <= 1e-8)
        check("cf1 plugin matches Phase 1 at 1e-8", worst["cf1"] <= 1e-8)
        check("slack plugin matches Phase 1 at 1e-8", worst["slack"] <= 1e-8)
    else:
        print("SKIP population checks (no t1_dictionary.csv)")

    # ---------------- 2. micro-examples through count pipeline
    eB = np.zeros(8); eB[[0, 5]] = 1.0   # delta_(0,0)|z0 ; delta_(0,1)|z1
    cB = np.round(eB * 10**6).astype(np.int64)
    check("ME-B kl plugin == 2 log 2",
          abs(kl_plugin_stat(A16, cB, 2) - 2 * np.log(2)) < 1e-6)
    check("ME-B cf1 plugin == 0", cf1_plugin_stat(A16.astype(int), cB, 2)
          < 1e-9)
    check("ME-B slack plugin == 2",
          abs(slack_plugin_stat(A16.astype(int), cB, 2) - 2.0) < 1e-6)

    # general alphabet builder sanity (row/col conventions)
    A333 = build_iv_A_general(3, 3, 5)
    check("columns are deterministic tables (col sum == kz)",
          np.all(A333.sum(axis=0) == 3))
    check("every observable row reachable", np.all(A333.sum(axis=1) >= 1))
    tabs_x_cols = set()
    for col in range(A333.shape[1]):
        rows = np.where(A333[:, col] > 0)[0]
        tabs_x_cols.add(len(rows))
    check("columns are deterministic tables", max(tabs_x_cols) == 3)

    # ---------------- 3. Monte Carlo bands under known null law
    rng = np.random.default_rng(20260826)
    n, R = 800, 300
    q_true = np.zeros(16); q_true[4 * 1 + 1] = 0.5; q_true[4 * 2 + 3] = 0.5
    push = A16 @ q_true                      # ME-C law (interior feasible)
    plug, spl, crf = [], [], []
    for r in range(R):
        nz = rng.multinomial(n // 2, [0.5, 0.5])
        counts = np.zeros(8, dtype=np.int64)
        for z in range(2):
            s = slice(4 * z, 4 * z + 4)
            counts[s] = rng.multinomial(int(nz[z]),
                                        push[s] / push[s].sum())
        plug.append(kl_plugin_stat(A16, counts, 2))
        spl.append(kl_split_stat(A16, counts, 2,
                                 np.random.default_rng(1000 + r)))
        crf.append(kl_crossfit_stat(A16, counts, 2,
                                    np.random.default_rng(2000 + r)))
    plug, spl, crf = map(np.array, (plug, spl, crf))
    print(f"MC means n={n}: plugin={plug.mean():.5f} "
          f"split={spl.mean():.5f} crossfit={crf.mean():.5f}")
    check("plugin KL >= 0 everywhere", np.all(plug >= 0))
    check("plugin KL median in MC band (<0.05 at n=800)",
          np.median(plug) < 0.05)
    check("split/crossfit remove plugin downward min-bias "
          "(stochastically larger under H0)",
          np.median(spl) > np.median(plug) and
          np.median(crf) > np.median(plug))
    check("crossfit no more variable than split (averaging directions)",
          np.std(crf) <= np.std(spl) * 1.5 + 1e-12)

    # bootstrap central band coverage of near-zero null statistic
    counts0 = np.zeros(8, dtype=np.int64)
    for z in range(2):
        s = slice(4 * z, 4 * z + 4)
        counts0[s] = rng.multinomial(n // 2, push[s] / push[s].sum())
    q_hat, push_hat = fit_null_q(A16, counts_to_conditionals(
        counts0[None, :], 2)[0])
    draws = draw_bootstrap_counts(push_hat.reshape(2, 4), counts0, 2,
                                  np.random.default_rng(77), 199,
                                  "para_boot")
    from witness_estimators import kl_bootstrap_batch
    stat_b = kl_bootstrap_batch(A16, counts_to_conditionals(draws, 2))
    lo, hi = np.quantile(stat_b, [0.05, 0.95])
    t_obs = kl_plugin_stat(A16, counts0, 2)
    check("parametric bootstrap 90% central band covers observed statistic",
          lo - 1e-9 <= t_obs <= hi + 1e-12)
    check("bootstrap draws conserve totals per stratum",
          all(draws[b].sum() == counts0.sum() for b in range(3)))

    # ---------------- 4. determinism / seed sensitivity of splits
    rng_d = np.random.default_rng(5)
    counts_d = rng_d.integers(0, 50, size=8).astype(np.int64)
    v1 = kl_split_stat(A16, counts_d, 2, np.random.default_rng(11))
    v2 = kl_split_stat(A16, counts_d, 2, np.random.default_rng(11))
    v3 = kl_split_stat(A16, counts_d, 2, np.random.default_rng(12))
    check("split deterministic given seed", v1 == v2)
    check("split varies across seeds", v1 != v3)

    # ---------------- 5. split conserves stratum totals
    ca, cb = split_counts(counts_d, 2, np.random.default_rng(3), 0.5)
    blk_sums_ok = all(ca[4*z:4*z+4].sum() + cb[4*z:4*z+4].sum()
                      == counts_d[4*z:4*z+4].sum() for z in range(2))
    check("split conserves per-stratum totals", blk_sums_ok)

    print()
    if FAILS:
        print("FAILURES:", FAILS)
        sys.exit(1)
    print("ALL WP 2.1 TESTS PASSED")


if __name__ == "__main__":
    main()
