"""WP 2.4 scaling pain map: runtime/memory vs alphabet geometry and n.

Measures per-call wall time and peak extra memory for every method on
synthetic datasets drawn from a random null SCM:
counts build, KL-EM plugin (single + B=100 batched), split/crossfit,
cf1_margin LP, slack LP, order-2 inflation feasibility LP, GM battery eval,
G-test. Also LP-only scaling across a Q x K sweep.

Output: results/phase2_discrete/lp_walltime_map.csv
"""
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from models import build_iv_A_general  # noqa: E402
from witnesses import kl_em_batch  # noqa: E402
from witness_estimators import (counts_to_conditionals,  # noqa: E402
                                kl_split_crossfit_stat, cf1_plugin_stat,
                                slack_plugin_stat)
from calibration import fit_null_q  # noqa: E402
import phase2_dgps as dg  # noqa: E402
from baselines.ineq_tests import (inflation_order2_feasible,  # noqa: E402
                                  build_gm_battery, gtest_independence_stat)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase2_discrete"


def timed(fn, *a, **k):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **k)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, peak, out


def bench_cell(kz, kx, ky, n, rep=3):
    M = build_iv_A_general(kz, kx, ky)
    Mint = M.astype(int)
    Wg, cg = get_battery_cached(M, (kz, kx, ky))
    rows = []
    for r in range(rep):
        rng = np.random.default_rng(31_000 + 17 * r)
        rows_dgp, meta = dg.sample_rows_null(rng, n, kz, kx, ky,
                                             "u_on_x")
        counts = dg.counts_from_rows(rows_dgp, kz, kx, ky)
        cond = counts_to_conditionals(counts[None, :], kz)[0]
        base = {"kz": kz, "kx": kx, "ky": ky, "K": kz * kx * ky,
                "Q": M.shape[1], "n": n, "rep": r}

        def add(name, fn, *a, **k):
            dt, mem, val = timed(fn, *a, **k)
            rows.append(dict(base, method=name, seconds=dt, peak_bytes=mem))

        add("counts_build", lambda: dg.counts_from_rows(rows_dgp, kz, kx, ky))
        add("kl_plugin_single", kl_em_batch, M.astype(float),
            cond[None, :])
        boot_cond = np.repeat(cond[None, :], 100, axis=0)
        add("kl_plugin_batch100", kl_em_batch, M.astype(float), boot_cond,
            n_iter=150, tol=1e-9)
        add("split_crossfit", kl_split_crossfit_stat, M,
            counts, kz, np.random.default_rng(r))
        add("cf1_lp", cf1_plugin_stat, Mint, counts, kz)
        add("slack_lp", slack_plugin_stat, Mint, counts, kz)
        add("null_fit_em", fit_null_q, M, cond)
        add("inflation_order2", inflation_order2_feasible, M, cond)
        add("gm_battery_eval", facet_eval, cond, Wg, cg)
        add("gtest", gtest_independence_stat, counts, kz, kx, ky)

        qh, push = fit_null_q(M, cond)
        from calibration import draw_bootstrap_counts
        add("bootstrap_draw_paraB99", draw_bootstrap_counts,
            push.reshape(kz, -1), counts, kz, np.random.default_rng(r), 99,
            "para_boot")
    return rows


def facet_eval(cond, Wg, cg):
    return float(np.max(np.maximum(np.asarray(cond) @ Wg.T - cg, 0.0)))


_BAT_CACHE = {}


def get_battery_cached(M, cell):
    key = tuple(cell)
    if key not in _BAT_CACHE:
        _BAT_CACHE[key] = get_gm_battery_shared(M, cell)
    return _BAT_CACHE[key]


def get_gm_battery_shared(M, cell):
    from run_wp22_calibration import get_gm_battery
    return get_gm_battery(M, cell)


def main():
    cfg = json.loads((ROOT / "configs" / "seeds.json").read_text())["phase2"]
    RES.mkdir(parents=True, exist_ok=True)
    rows = []
    for kz, kx, ky in [tuple(c) for c in cfg["alphabet_cells"]]:
        for n in (500, 2000, 8000):
            t0 = time.time()
            rows += bench_cell(kz, kx, ky, n)
            print(f"[wp24] ({kz},{kx},{ky}) n={n}: {time.time()-t0:.1f}s",
                  flush=True)
    df = pd.DataFrame(rows)

    # LP-only synthetic sweep over (Q, K): random polytope instances of the
    # IV geometry are covered above; here isolate solver scaling.
    sweep_rows = []
    for (kx, ky) in [(2, 2), (2, 5), (3, 5), (2, 8)]:
        for kz in (2, 3, 4):
            A = build_iv_A_general(kz, kx, ky)
            Mint = A.astype(int)
            K, Q = A.shape
            for r in range(3):
                rng = np.random.default_rng(77_000 + r)
                cond = np.concatenate(
                    [rng.dirichlet(np.ones(K // kz)) for _ in range(kz)])
                dt1, m1, _ = timed(cf1_plugin_stat, Mint,
                                   np.round(cond * 4000), kz)
                dt2, m2, _ = timed(slack_plugin_stat, Mint,
                                   np.round(cond * 4000), kz)
                sweep_rows.append({"kz": kz, "kx": kx, "ky": ky, "K": K,
                                   "Q": Q, "method": "cf1_lp",
                                   "seconds": dt1, "peak_bytes": m1})
                sweep_rows.append({"kz": kz, "kx": kx, "ky": ky, "K": K,
                                   "Q": Q, "method": "slack_lp",
                                   "seconds": dt2, "peak_bytes": m2})
    df = pd.concat([df, pd.DataFrame(sweep_rows)], ignore_index=True)
    df.to_csv(RES / "lp_walltime_map.csv", index=False)
    print(f"wrote lp_walltime_map.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
