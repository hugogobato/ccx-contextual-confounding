"""Phase 4 Block D: scaling study (runtime/memory vs (n, d, k)).

Discrete: per-dataset statistic-evaluation time and LP solve time for the
Phase 4 cells (2-2-4/5/8) at n {500,2000,8000}, plus the synthetic LP-only
(Q,) sweep reused from WP 2.4.
Continuous: k1_v3 / k2_v3 / hsic_resid point-statistic wall time and
tracemalloc peak vs n {2000..50000} at d=2.

Output: results/raw/phase4/p4scale.csv
"""
import argparse
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
from run_wp22_calibration import observed_stats, get_gm_battery  # noqa: E402
from witness_estimators import cf1_plugin_stat, slack_plugin_stat  # noqa: E402
from continuous_witness import k1_v3_stat, k2_v3_stat, hsic_resid_stat  # noqa: E402
import phase2_dgps as dg  # noqa: E402
import phase3_dgps as d3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase4"
P4 = None


def timed(fn, *a, **k):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **k)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, peak, out


def discrete_rows(p4):
    rows = []
    for cell in p4["scaling"]["discrete"]["cells"]:
        kz, kx, ky = cell
        cell_id = "-".join(map(str, cell))
        M = build_iv_A_general(kz, kx, ky)
        Mint = M.astype(int)
        Wg, cg = get_gm_battery(M, (kz, kx, ky))
        mix = dg.MixtureCell(kz, kx, ky)
        for n in p4["scaling"]["discrete"]["n_grid"]:
            for r in range(p4["scaling"]["discrete"]["reps"]):
                rng = np.random.default_rng(880000 + r)
                data = mix.sample_rows(rng, n, 0.4)
                counts = dg.counts_from_rows(data, kz, kx, ky)
                dt, peak, _ = timed(observed_stats, M, Mint, counts, kz,
                                    kx, ky, None, None, Wg, cg)
                rows.append({"arm": "discrete", "cell": cell_id, "Q": M.shape[1],
                             "n": n, "rep": r, "method": "stats_all",
                             "seconds": dt, "peak_bytes": peak})
                for meth, fn in (("cf1_lp", cf1_plugin_stat),
                                 ("slack_lp", slack_plugin_stat)):
                    dt, peak, _ = timed(fn, Mint, counts, kz)
                    rows.append({"arm": "discrete", "cell": cell_id,
                                 "Q": M.shape[1], "n": n, "rep": r,
                                 "method": meth, "seconds": dt,
                                 "peak_bytes": peak})
    return rows


def continuous_rows(p4):
    rows = []
    d = p4["scaling"]["continuous"]["d"]
    for n in p4["scaling"]["continuous"]["n_grid"]:
        for r in range(p4["scaling"]["continuous"]["reps"]):
            rng = np.random.default_rng(890000 + r)
            x, y, _ = d3.sample_confounded(rng, n, d, 0.4, noise="gauss",
                                           kind="conf_nonlin")
            for meth, fn in (("k1_v3", k1_v3_stat), ("k2_v3", k2_v3_stat),
                             ("hsic_resid", hsic_resid_stat)):
                if meth == "hsic_resid":
                    dt, peak, _ = timed(fn, x[:800], y[:800])
                else:
                    dt, peak, _ = timed(fn, x, y, trim_q=0.05)
                rows.append({"arm": "continuous", "cell": f"d{d}",
                             "Q": n, "n": n, "rep": r, "method": meth,
                             "seconds": dt, "peak_bytes": peak})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["discrete", "continuous"],
                    default=None)
    args = ap.parse_args()
    p4 = P4 or __import__("json").loads(
        (ROOT / "configs" / "phase4.json").read_text())
    rows = []
    if args.arm in (None, "discrete"):
        rows += discrete_rows(p4)
        print("discrete done", flush=True)
    if args.arm in (None, "continuous"):
        rows += continuous_rows(p4)
        print("continuous done", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "p4scale.csv"
    old = pd.read_csv(out) if out.exists() else None
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
