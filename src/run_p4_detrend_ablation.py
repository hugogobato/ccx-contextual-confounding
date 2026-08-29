"""Phase 4 L4-remedy ablation: per-context QUADRATIC vs LINEAR detrend.

Question (user decision 2026-08-29): does higher-order detrending rescue
the B4 heavy-curvature adversarial config without breaking size elsewhere
or killing the conf_nonlin/t3 flagship power? If yes -> adopt as the
paper variant (full re-run to follow); if no -> document B4 as a declared
limitation of the linear-detrend surrogate.

Focused cells (DIAGNOSTIC, not part of the frozen Gate D inputs):
- adversarial : B1-B4 at n=5000 (100 seeds) — FPR per detrend
- null sizes  : {null_gauss, null_nonparam} x {gauss, t3} x n
                {2000, 8000, 20000} x d=2 (100 seeds)
- power       : conf_nonlin x t3 x n {2000, 20000} x b {0.4, 0.8} x d=2
                (100 seeds)

Rows -> results/raw/phase4/p4detrend_*.csv
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

from continuous_witness import k1_k2_perm_calibration  # noqa: E402
from calibration import critical_values  # noqa: E402
import phase3_dgps as d3  # noqa: E402
from run_p4_continuous import _adv_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase4"
ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]
TRIM = 0.05
B = 99
NSEEDS = 100
DETS = ("linear", "quadratic")


def process(g):
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        if g["block"] == "adversarial":
            x, y = _adv_sample(rng, g["n"], {"id": g["cfg"], "n": g["n"],
                                             "d": 2, "noise": g["noise"]})
        elif g["block"] == "null":
            x, y, _ = d3.sample_null(rng, g["n"], 2, noise=g["noise"],
                                     kind=g["cfg"])
        else:
            x, y, _ = d3.sample_confounded(rng, g["n"], 2, g["b"],
                                           noise=g["noise"],
                                           kind=g["cfg"])
        for det in DETS:
            obs, dr = k1_k2_perm_calibration(
                x, y, B=B, trims=(TRIM,), detrend=det,
                rng=np.random.default_rng(seed + 7500000 + hash(det) % 1000))
            for meth in ("k1", "k2"):
                cvs = critical_values(dr[TRIM][meth], ALPHA_GRID)
                row = {"block": g["block"], "cfg": g["cfg"],
                       "noise": g["noise"], "n": g["n"], "d": 2,
                       "b": g.get("b", np.nan), "seed": seed,
                       "method": meth, "detrend": det, "trim": TRIM,
                       "B": B, "stat_obs": obs[TRIM][meth]}
                for a in ALPHA_GRID:
                    row[f"cv_{a:.2f}"] = cvs[a]
                out.append(row)
    return out


def make_groups(pilot=False):
    seeds = list(range(900000, 900000 + (6 if pilot else NSEEDS)))
    groups = []
    p4 = json.loads((ROOT / "configs" / "phase4.json").read_text())
    specs = p4["adversarial_nulls"]["continuous"]
    for s in specs:                                   # adversarial B1-B4
        groups.append({"block": "adversarial", "cfg": s["id"], "n": s["n"],
                       "noise": s["noise"], "seeds": seeds})
    for noise in ("gauss", "t3"):                     # null sizes
        for kind in ("null_gauss", "null_nonparam"):
            for n in ([2000] if pilot else (2000, 8000, 20000)):
                groups.append({"block": "null", "cfg": kind, "n": n,
                               "noise": noise, "seeds": seeds})
    for n in ([2000] if pilot else (2000, 20000)):    # power
        for b in (0.4, 0.8):
            groups.append({"block": "alt", "cfg": "conf_nonlin", "n": n,
                           "noise": "t3", "b": b, "seeds": seeds})
    return groups


_JOB = {}


def _run_idx(i):
    g, proc, fn_of = _JOB["groups"][i], _JOB["proc"], _JOB["fn_of"]
    return run_one(fn_of(g), g, proc)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--one", type=int, default=-1)
    args = ap.parse_args()
    groups = make_groups(pilot=args.pilot)
    fn_of = lambda g: (RAW / (f"p4detrend_{g['block']}_{g['cfg']}_"
                              f"{g['noise']}_n{g['n']}"
                              + (f"_b{g['b']:.1f}" if g["block"] == "alt"
                                 else "") + ".csv"))
    if args.one >= 0:
        print(run_one(fn_of(groups[args.one]), groups[args.one], process),
              flush=True)
        return
    _JOB.update(groups=groups, proc=process, fn_of=fn_of)
    print(f"[p4detrend] groups: {len(groups)}", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    done = 0
    with mp.Pool(processes=int(os.environ.get("CCX_WORKERS", "6")),
                 maxtasksperchild=2) as pool:
        for nr in pool.imap_unordered(_run_idx, range(len(groups)),
                                      chunksize=1):
            done += 1
            print(f"[p4detrend] {done}/{len(groups)} rows={nr} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    print("driver complete")


if __name__ == "__main__":
    main()
