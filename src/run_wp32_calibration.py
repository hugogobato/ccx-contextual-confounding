"""WP 3.2 null calibration + crackle stress for the continuous witnesses.

Configs: n in {500,2000,8000} x d in {2,3,5} x noise {gauss,t3,lognorm}
x null kind {null_gauss, null_nonparam}; seeds 200 (pilot: 20).
Methods: k1, k2, hsic. Calibration (v3): per-context linear detrend +
winsorization (primary trim 0.05) + affine standardization + context-
label permutation B=199; HSIC baseline uses pairs-bootstrap CVs.

Rows -> results/raw/phase3/wp32_*.csv; aggregated by aggregate_phase3.py
into null_calibration.csv and crackle_stress.csv.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from continuous_witness import (k1_k2_perm_calibration,   # noqa: E402
                                hsic_resid_stat, hsic_pairs_bootstrap)
from calibration import critical_values  # noqa: E402
import phase3_dgps as d3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase3"

ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]
HSIC_CAP = 800           # pairs-bootstrap resample cap


def process_group(g):
    n, d, noise, kind = g["n"], g["d"], g["noise"], g["kind"]
    B = g["B"]
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        x, y, _W = d3.sample_null(rng, n, d, noise=noise, kind=kind)

        t0 = time.perf_counter()
        # v3: primary trim 0.05 at full B; sensitivity trims on
        # gauss-noise nulls only, at reduced draw budgets.
        if noise == "gauss":
            trims = (0.0, 0.01, 0.05)
            bmap = {0.0: min(B, 49), 0.01: min(B, 49), 0.05: B}
        else:
            trims = (0.05,)
            bmap = None
        obs, dr = k1_k2_perm_calibration(
            x, y, B=B, trims=trims, bmap=bmap,
            rng=np.random.default_rng(seed + 7000000))
        xc, yc = x[:HSIC_CAP], y[:HSIC_CAP]
        obs_hsic = abs(hsic_resid_stat(xc, yc))
        hb = hsic_pairs_bootstrap(xc, yc, B=B,
                                  rng=np.random.default_rng(
                                      seed + 7200000), cap=HSIC_CAP)
        dt_boot = time.perf_counter() - t0

        base = {"n": n, "d": d, "noise": noise, "kind": kind, "seed": seed}
        for meth in ("k1", "k2"):
            for tq in trims:
                cvs = critical_values(dr[tq][meth], ALPHA_GRID)
                row = dict(base, method=meth, trim=tq,
                           B=len(dr[tq][meth]),
                           stat_obs=obs[tq][meth], dt_boot_s=dt_boot)
                for a in ALPHA_GRID:
                    row[f"cv_{a:.2f}"] = cvs[a]
                out.append(row)
        cvs = critical_values(hb, ALPHA_GRID)
        row = dict(base, method="hsic", trim=0.0,
                   B=len(hb), stat_obs=obs_hsic,
                   dt_boot_s=dt_boot)
        for a in ALPHA_GRID:
            row[f"cv_{a:.2f}"] = cvs[a]
        out.append(row)
    return out


def make_groups(cfg_like, pilot=False):
    groups = []
    ns = [500, 2000, 8000]
    ds = [2, 3, 5]
    noises = ["gauss", "t3", "lognorm"]
    kinds = ["null_gauss", "null_nonparam"]
    B = 49 if pilot else 199
    nseeds = 12 if pilot else json.loads(
        (ROOT / "configs" / "seeds.json").read_text())["phase2"].get("wp32_seeds", 200)
    for nv in ([500, 2000] if pilot else ns):
        for dd in ds:
            for nz in noises:
                for kd in kinds:
                    groups.append({"n": nv, "d": dd, "noise": nz,
                                   "kind": kd, "B": B,
                                   "seeds": list(range(400000,
                                                       400000 + nseeds))})
    return groups


def run_one(g):
    fn = RAW / f"wp32_n{g['n']}_d{g['d']}_{g['noise']}_{g['kind']}.csv"
    if fn.exists():
        return -1
    rows = process_group(g)
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = str(fn) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, fn)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dump-groups", action="store_true")
    ap.add_argument("--one", type=int, default=-1)
    args = ap.parse_args()
    cfg_like = {}
    groups = make_groups(cfg_like, pilot=args.pilot)
    if args.dump_groups:
        (ROOT / "configs" / "phase3_groups.json").write_text(
            json.dumps(groups))
        print(f"wrote {len(groups)} wp32 groups")
        return
    if args.one >= 0:
        print(run_one(groups[args.one]), flush=True)
        return
    print(f"wp32 groups: {len(groups)}")
    import multiprocessing as mp
    with mp.Pool(processes=int(os.environ.get("CCX_WORKERS", "6")),
                 maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, groups)):
            print(f"[wp32] {i+1}/{len(groups)} rows={r}", flush=True)


if __name__ == "__main__":
    main()
