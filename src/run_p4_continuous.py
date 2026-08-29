"""Phase 4 Block B/C: continuous ladder (n=20000 matched nulls, n=8000
alternatives, adversarial nulls).

Arms (configs/phase4.json):
- null : n=20000 x d {2,3,5} x noise {gauss,t3} x kind {null_gauss,
         null_nonparam}; seeds 400000-400199 (WP 3.2 stream); v3 pipeline,
         B=199, trim policy identical to run_wp32_calibration. Fills the
         only missing matched-null cells of the Phase 4 grid (D-P4.2).
         Raw rows -> p4cnull_*.csv (schema = wp32).
- alt  : n=8000 x d {2,3,5} x noise {gauss,t3} x kinds {conf_lin,
         conf_nonlin} x b {0.2,0.4,0.6,0.8}; seeds 500000-500199 (WP 3.3
         stream); point statistics. All other grid cells reuse verified
         WP 3.3 raw data (D-P4.2). Raw rows -> p4calt_*.csv (schema = wp33).
- adv  : 4 adversarial null configs (B1-B4), self-calibrated per dataset
         (k1/k2 label-permutation B=99 at trim 0.05; hsic pairs-bootstrap
         B=99). Raw rows -> p4cadv_*.csv.
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

from continuous_witness import (k1_k2_perm_calibration, k1_v3_stat,   # noqa: E402
                                k2_v3_stat, hsic_resid_stat,
                                hsic_pairs_bootstrap)
from calibration import critical_values  # noqa: E402
import phase3_dgps as d3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase4"
ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]
HSIC_CAP = 800
TRIM_PRIMARY = 0.05
DETREND = "quadratic"   # v4 variant (D-P4.9)


# --------------------------------------------------------------- null arm

def process_null_group(g):
    n, d, noise, kind = g["n"], g["d"], g["noise"], g["kind"]
    B = g["B"]
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        x, y, _W = d3.sample_null(rng, n, d, noise=noise, kind=kind)
        t0 = time.perf_counter()
        if noise == "gauss":
            trims = (0.0, 0.01, 0.05)
            bmap = {0.0: min(B, 49), 0.01: min(B, 49), 0.05: B}
        else:
            trims = (0.05,)
            bmap = None
        obs, dr = k1_k2_perm_calibration(
            x, y, B=B, trims=trims, bmap=bmap, detrend=DETREND,
            rng=np.random.default_rng(seed + 7000000))
        xc, yc = x[:HSIC_CAP], y[:HSIC_CAP]
        obs_hsic = abs(hsic_resid_stat(xc, yc))
        hb = hsic_pairs_bootstrap(xc, yc, B=B,
                                  rng=np.random.default_rng(seed + 7200000),
                                  cap=HSIC_CAP)
        dt_boot = time.perf_counter() - t0
        base = {"n": n, "d": d, "noise": noise, "kind": kind, "seed": seed}
        for meth in ("k1", "k2"):
            for tq in trims:
                cvs = critical_values(dr[tq][meth], ALPHA_GRID)
                row = dict(base, method=meth, trim=tq, B=len(dr[tq][meth]),
                           stat_obs=obs[tq][meth], dt_boot_s=dt_boot)
                for a in ALPHA_GRID:
                    row[f"cv_{a:.2f}"] = cvs[a]
                out.append(row)
        cvs = critical_values(hb, ALPHA_GRID)
        row = dict(base, method="hsic", trim=0.0, B=len(hb),
                   stat_obs=obs_hsic, dt_boot_s=dt_boot)
        for a in ALPHA_GRID:
            row[f"cv_{a:.2f}"] = cvs[a]
        out.append(row)
    return out


def make_null_groups(p4, pilot=False):
    cg = p4["continuous_grid"]
    nseeds = 12 if pilot else cg["reps"]
    seeds = list(range(400000, 400000 + nseeds))
    groups = []
    # v4 (D-P4.9): the quadratic detrend changes values at every n, so
    # matched nulls are required (and run) at the full grid.
    for n in ([20000] if pilot else [2000, 8000, 20000]):
        for d in cg["cells_d"]:
            for noise in cg["noise"]:
                for kind in cg["null_kinds"]:
                    groups.append({"n": n, "d": d, "noise": noise,
                                   "kind": kind, "B": 49 if pilot else 199,
                                   "seeds": seeds})
    return groups


# ---------------------------------------------------------------- alt arm

def process_alt_group(g):
    n, d, noise, kind, b = (g["n"], g["d"], g["noise"], g["kind"], g["b"])
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        x, y, _W = d3.sample_confounded(rng, n, d, b, noise=noise,
                                        kind=kind)
        obs = {"k1": k1_v3_stat(x, y, trim_q=TRIM_PRIMARY,
                                detrend=DETREND),
               "k2": k2_v3_stat(x, y, trim_q=TRIM_PRIMARY,
                                detrend=DETREND),
               "hsic": abs(hsic_resid_stat(x[:HSIC_CAP], y[:HSIC_CAP]))}
        for meth, sv in obs.items():
            out.append({"n": n, "d": d, "noise": noise, "kind": kind,
                        "b": b, "seed": seed, "method": meth,
                        "stat_obs": sv})
    return out


def make_alt_groups(p4, pilot=False):
    cg = p4["continuous_grid"]
    nseeds = 12 if pilot else cg["reps"]
    seeds = list(range(500000, 500000 + nseeds))
    groups = []
    # v4 (D-P4.9): alternatives at the full grid (WP3.3 reuse retired).
    for n in ([8000] if pilot else [2000, 8000, 20000]):
        for d in cg["cells_d"]:
            for noise in cg["noise"]:
                for kind in cg["alt_kinds"]:
                    for b in (cg["b_grid"][:2] if pilot
                              else cg["b_grid"]):
                        groups.append({"n": n, "d": d, "noise": noise,
                                       "kind": kind, "b": b, "seeds": seeds})
    return groups


# --------------------------------------------------------- adversarial arm

def _adv_sample(rng, n, spec):
    """Adversarial null DGPs B1-B4 (configs/phase4.json). All satisfy the
    declared shared-shape ANM class; they stress selection, design
    multimodality, residual atoms, and curvature."""
    aid = spec["id"]
    if aid == "B1_selection_trunc":
        x, y, _ = d3.sample_null(rng, int(n * 1.6), 2, noise="gauss",
                                 kind="null_gauss")
        keep = np.abs(x) <= np.quantile(np.abs(x), 0.8)
        x, y = x[keep][:n], y[keep][:n]
        return x, y
    if aid == "B2_mixture_x":
        comp = rng.random(n) < 0.5
        x = np.where(comp, rng.normal(-2, 1, n), rng.normal(2, 1, n))
        eps = d3._noise(rng, n, "t3")
        y = 0.8 * np.tanh(x) + eps
        return x, y
    if aid == "B3_clipped_noise":
        x, _, _ = d3.sample_null(rng, n, 2, noise="gauss",
                                 kind="null_nonparam")
        eps = d3._noise(rng, n, "t3")
        lo, hi = np.quantile(eps, [0.10, 0.90])
        y = 0.8 * np.tanh(x) + np.clip(eps, lo, hi)
        return x, y
    if aid == "B4_heavy_curvature":
        x, _, _ = d3.sample_null(rng, n, 2, noise="gauss",
                                 kind="null_gauss")
        x = np.sort(x)                     # deterministic design stress
        eps = d3._noise(rng, n, "t3")
        y = 2.5 * np.sin(1.3 * x) + 0.15 * x ** 2 + eps
        return x, y
    raise ValueError(aid)


def process_adv_group(g):
    spec, B = g["spec"], g["B"]
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        x, y = _adv_sample(rng, spec["n"], spec)
        obs, dr = k1_k2_perm_calibration(
            x, y, B=B, trims=(TRIM_PRIMARY,), detrend=DETREND,
            rng=np.random.default_rng(seed + 7300000))
        xc, yc = x[:HSIC_CAP], y[:HSIC_CAP]
        obs_hsic = abs(hsic_resid_stat(xc, yc))
        hb = hsic_pairs_bootstrap(xc, yc, B=B,
                                  rng=np.random.default_rng(seed + 7400000),
                                  cap=HSIC_CAP)
        base = {"adv_id": spec["id"], "n": spec["n"], "d": spec["d"],
                "noise": spec["noise"], "seed": seed}
        for meth in ("k1", "k2"):
            cvs = critical_values(dr[TRIM_PRIMARY][meth], ALPHA_GRID)
            row = dict(base, method=meth, trim=TRIM_PRIMARY,
                       B=len(dr[TRIM_PRIMARY][meth]),
                       stat_obs=obs[TRIM_PRIMARY][meth])
            for a in ALPHA_GRID:
                row[f"cv_{a:.2f}"] = cvs[a]
            out.append(row)
        cvs = critical_values(hb, ALPHA_GRID)
        row = dict(base, method="hsic", trim=0.0, B=len(hb),
                   stat_obs=obs_hsic)
        for a in ALPHA_GRID:
            row[f"cv_{a:.2f}"] = cvs[a]
        out.append(row)
    return out


def make_adv_groups(p4, pilot=False):
    nseeds = 12 if pilot else p4["adversarial_nulls"]["reps"]
    seeds = list(range(900000, 900000 + nseeds))
    cfg4 = json.loads((ROOT / "configs" / "seeds.json").read_text())["phase4"]
    B = cfg4["adversarial_perm_B"]
    return [{"spec": s, "B": B, "seeds": seeds}
            for s in p4["adversarial_nulls"]["continuous"]]


# ------------------------------------------------------------------ driver

_JOB = {}


def run_one(fn, g, proc):
    if fn.exists():
        return -1
    rows = proc(g)
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = str(fn) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, fn)
    return len(rows)


def _run_idx(i):
    g, proc, fn_of = _JOB["groups"][i], _JOB["proc"], _JOB["fn_of"]
    return run_one(fn_of(g), g, proc)


def run_all(tag, groups, proc, fn_of):
    _JOB.update(groups=groups, proc=proc, fn_of=fn_of)
    print(f"[{tag}] groups: {len(groups)}", flush=True)
    import multiprocessing as mp
    nw = int(os.environ.get("CCX_WORKERS", "6"))
    t0 = time.time()
    done = 0
    with mp.Pool(processes=nw, maxtasksperchild=2) as pool:
        for nr in pool.imap_unordered(_run_idx, range(len(groups)),
                                      chunksize=1):
            done += 1
            print(f"[{tag}] {done}/{len(groups)} rows={nr} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    print("driver complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["null", "alt", "adv"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--one", type=int, default=-1)
    ap.add_argument("--dump-groups", action="store_true")
    args = ap.parse_args()
    p4 = json.loads((ROOT / "configs" / "phase4.json").read_text())
    if args.arm == "null":
        groups = make_null_groups(p4, pilot=args.pilot)
        proc, tag = process_null_group, "p4cnull"
        fn_of = lambda g: (RAW / f"p4cnull_n{g['n']}_d{g['d']}_"
                               f"{g['noise']}_{g['kind']}.csv")
    elif args.arm == "alt":
        groups = make_alt_groups(p4, pilot=args.pilot)
        proc, tag = process_alt_group, "p4calt"
        fn_of = lambda g: (RAW / f"p4calt_n{g['n']}_d{g['d']}_{g['noise']}_"
                               f"{g['kind']}_b{g['b']:.1f}.csv")
    else:
        groups = make_adv_groups(p4, pilot=args.pilot)
        proc, tag = process_adv_group, "p4cadv"
        fn_of = lambda g: RAW / f"p4cadv_{g['spec']['id']}.csv"

    if args.dump_groups:
        print(f"[{tag}] {len(groups)} groups")
        return
    if args.one >= 0:
        print(run_one(fn_of(groups[args.one]), groups[args.one], proc),
              flush=True)
        return
    run_all(tag, groups, proc, fn_of)


if __name__ == "__main__":
    main()
