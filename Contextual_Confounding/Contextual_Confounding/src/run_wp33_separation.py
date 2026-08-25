"""WP 3.3 decisive separation study (B2 section 2.2 regime).

Alternatives: conf_lin / conf_nonlin with strength b in {0.1..0.9};
n in {500,2000,5000,10000,20000}; d in {2,3,5}; noise {gauss,t3,lognorm}.
Critical values: transferred from WP 3.2 null runs at matching
(n, d, noise) via pooled CVs. Methods: k1/k2/hsic at predeclared trimming.

Pass rule (predeclared): witness power >= size + 0.25 at b >= 0.4,
n <= 5000, d <= 3, size <= 0.06 at alpha = 0.05, in >= 2 of 3 favorable
cells where ALL baselines remain <= size + 0.10; power monotone in b.
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

from continuous_witness import k1_witness, k2_witness, hsic_stat  # noqa: E402
import phase3_dgps as d3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase3"
ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]
TRIM_PRIMARY = 0.01       # primary policy per plan (trim 1st/99th)
HSIC_CAP = 2500


def process_group(g):
    n, d, noise, kind, b = g["n"], g["d"], g["noise"], g["kind"], g["b"]
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        x, y, _W = d3.sample_confounded(rng, n, d, b, noise=noise,
                                        kind=kind)
        obs = {"k1": k1_witness(x, y, trim_q=TRIM_PRIMARY),
               "k2": k2_witness(x, y, trim_q=TRIM_PRIMARY),
               "hsic": hsic_stat(x[:HSIC_CAP], y[:HSIC_CAP])}
        for meth, sv in obs.items():
            out.append({"n": n, "d": d, "noise": noise, "kind": kind,
                        "b": b, "seed": seed, "method": meth,
                        "stat_obs": sv})
    return out


def make_groups(pilot=False):
    groups = []
    bs = [0.2, 0.5, 0.8] if pilot else [round(0.1 * k, 1)
                                        for k in range(1, 10)]
    ns = [500, 2000] if pilot else [500, 2000, 5000, 10000, 20000]
    ds = [2, 3] if pilot else [2, 3, 5]
    noises = ["gauss"] if pilot else ["gauss", "t3", "lognorm"]
    kinds = ["conf_lin"] if pilot else ["conf_lin", "conf_nonlin"]
    nseeds = 12 if pilot else 200
    seeds = list(range(500000, 500000 + nseeds))
    for nv in ns:
        for dd in ds:
            for nz in noises:
                for kd in kinds:
                    for bv in bs:
                        groups.append({"n": nv, "d": dd, "noise": nz,
                                       "kind": kd, "b": bv, "seeds": seeds})
    return groups


def run_one(g):
    fn = RAW / (f"wp33_n{g['n']}_d{g['d']}_{g['noise']}_{g['kind']}_"
                f"b{g['b']:.1f}.csv")
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
    groups = make_groups(pilot=args.pilot)
    if args.dump_groups:
        (ROOT / "configs" / "phase3_sep_groups.json").write_text(
            json.dumps(groups))
        print(f"wrote {len(groups)} wp33 groups")
        return
    if args.one >= 0:
        print(run_one(groups[args.one]), flush=True)
        return
    print(f"wp33 groups: {len(groups)}")
    import multiprocessing as mp
    with mp.Pool(processes=int(os.environ.get("CCX_WORKERS", "6")),
                 maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, groups)):
            print(f"[wp33] {i+1}/{len(groups)} rows={r}", flush=True)


if __name__ == "__main__":
    main()
