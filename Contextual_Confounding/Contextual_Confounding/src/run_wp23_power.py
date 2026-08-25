"""WP 2.3 power study driver (mixture + mechanistic alternative families).

Critical values: pooled null CVs from WP 2.2 (results/phase2_discrete/
null_critical_values.csv), matched per (cell, n, stat, engine, alpha).
Alternative datasets need only point statistics (fast path).

Families:
- mixture    : response-space mixture with confounded-mass strength rho
               (all 7 cells x rho grid).
- mechanistic: logistic-threshold shared-U binary SCM (binary cell).

Outputs raw rows -> results/raw/phase2/wp23_*.csv; aggregated by
aggregate_wp23.py into power_curves.csv + runtime_scaling.csv.
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

from models import build_iv_A_general  # noqa: E402
from run_wp22_calibration import (load_cfg, get_gm_battery,  # noqa: E402
                                  get_binary_facets, observed_stats)
import phase2_dgps as dg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase2_discrete"
RAW = ROOT / "results" / "raw" / "phase2"
ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]

_CELL_CACHE = {}


def cell_objects(cell_id):
    if cell_id in _CELL_CACHE:
        return _CELL_CACHE[cell_id]
    kz, kx, ky = map(int, cell_id.split("-"))
    M = build_iv_A_general(kz, kx, ky)
    facW = facC = None
    if (kz, kx, ky) == (2, 2, 2):
        facW, facC = get_binary_facets()
    Wg, cg = get_gm_battery(M, (kz, kx, ky))
    mix = dg.MixtureCell(kz, kx, ky)
    _CELL_CACHE[cell_id] = (M, M.astype(int), kz, kx, ky,
                            facW, facC, Wg, cg, mix)
    return _CELL_CACHE[cell_id]


def process_group(group):
    cfg = group["cfg"]
    cell_id = "-".join(map(str, group["cell"]))
    M, Mint, kz, kx, ky, facW, facC, Wg, cg, mix = cell_objects(cell_id)
    n, rho, family = group["n"], group["rho"], group["family"]

    out = []
    for seed in group["seeds"]:
        rng = np.random.default_rng(seed)
        if family == "mixture":
            rows = mix.sample_rows(rng, n, rho)
        elif family == "mechanistic":
            assert (kz, kx, ky) == (2, 2, 2)
            rows = dg.mechanistic_binary_rows(rng, n, rho)
        else:
            raise ValueError(family)
        counts = dg.counts_from_rows(rows, kz, kx, ky)
        t0 = time.perf_counter()
        obs = observed_stats(M, Mint, counts, kz, kx, ky, facW, facC, Wg, cg)
        dt = time.perf_counter() - t0
        base = {"cell": cell_id, "family": family, "rho": rho, "n": n,
                "seed": seed, "dt_stats_s": dt}
        for statname, sv in obs.items():
            for eng in ("para_boot", "crt_cond", "subsample"):
                out.append(dict(base, stat=statname, engine=eng,
                                stat_obs=sv))
    return out


def make_groups(cfg, families, pilot=False):
    groups = []
    alt = cfg["alt_mixture_seeds"]
    mech = cfg["alt_mechanistic_seeds"]
    rhos = ([0.2, 0.4, 0.6] if pilot else cfg["rho_grid"])
    ns = ([250, 2000, 8000] if pilot else cfg["n_grid"])
    for cell in [tuple(c) for c in cfg["alphabet_cells"]]:
        cid = "-".join(map(str, cell))
        seeds_mix = list(range(alt["start"], alt["start"] + alt["count"]))
        if pilot:
            seeds_mix = seeds_mix[:12]
        for rho in rhos:
            for n in ns:
                groups.append({"cell": cell, "family": "mixture",
                               "rho": rho, "n": n, "seeds": seeds_mix,
                               "cfg": cfg})
    if "mechanistic" in families:
        seeds_m = list(range(mech["start"], mech["start"] + mech["count"]))
        if pilot:
            seeds_m = seeds_m[:12]
        for rho in rhos:
            for n in ns:
                groups.append({"cell": (2, 2, 2), "family": "mechanistic",
                               "rho": rho, "n": n, "seeds": seeds_m,
                               "cfg": cfg})
    return groups


def run_one_group(g):
    rows = process_group(g)
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    fn = RAW / (f"wp23_{'-'.join(map(str, g['cell']))}_{g['family']}_"
                f"rho{g['rho']:.1f}_n{g['n']}.csv")
    df.to_csv(fn, index=False)
    return g["cell"], g["family"], g["rho"], g["n"], len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--families", default="mixture,mechanistic")
    ap.add_argument("--dump-groups", action="store_true")
    ap.add_argument("--one", type=int, default=-1)
    args = ap.parse_args()
    cfg = load_cfg()
    groups = make_groups(cfg, args.families.split(","), pilot=args.pilot)

    if args.dump_groups:
        serial = [{**g, "cell": list(g["cell"])} for g in groups]
        (ROOT / "configs" / "phase2_power_groups.json").write_text(
            json.dumps(serial))
        print(f"wrote {len(groups)} power groups")
        return
    if args.one >= 0:
        print(run_one_group(groups[args.one]), flush=True)
        return

    print(f"groups: {len(groups)} "
          f"(datasets {sum(len(g['seeds']) for g in groups)})", flush=True)
    t0 = time.time()
    import multiprocessing as mp
    nw = int(os.environ.get("CCX_WORKERS", "6"))
    done = 0
    with mp.Pool(processes=nw, maxtasksperchild=4) as pool:
        for cell, fam, rho, n, nr in pool.imap_unordered(run_one_group,
                                                         groups, chunksize=1):
            done += 1
            el = time.time() - t0
            print(f"[wp23] {done}/{len(groups)} "
                  f"{'-'.join(map(str, cell))} {fam} rho={rho} n={n}: "
                  f"{nr} rows | {el:.0f}s", flush=True)
    print("driver complete")


if __name__ == "__main__":
    main()
