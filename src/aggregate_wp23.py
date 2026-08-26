"""WP 2.3 aggregator v2: matched-size power curves (predeclared metric).

For every alpha on the grid, merges per-kind-envelope critical values
(with conservative nearest-n transfer where the power grid lacks matched
null bootstraps), computes power(size) pairs per method, and interpolates
power at achieved-size targets {0.04, 0.06}.

Outputs: results/phase2_discrete/power_curves.csv
         results/phase2_discrete/runtime_scaling.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase2_discrete"
RAW = ROOT / "results" / "raw" / "phase2"

SIZE_TARGETS = [0.04, 0.06]
ALPHAS = [round(0.01 * a, 2) for a in range(1, 21)]


def main():
    files = sorted(RAW.glob("wp23_*.csv"))
    assert files, "no wp23 raw files"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    env = pd.read_csv(RES / "null_critical_values_envelope.csv")
    env["alpha_r"] = env["alpha"].round(2)
    lut = {(r.cell, r.n, r.stat, r.engine, r.alpha_r): r.cv_env
           for r in env.itertuples()}
    ns_by_key = {}
    for (c, n, s, e, a) in lut:
        ns_by_key.setdefault((c, s, e), set()).add(n)

    def cv_for(cell, n, stat, engine, a):
        if (cell, n, stat, engine, a) in lut:
            return lut[(cell, n, stat, engine, a)]
        cand_ns = ns_by_key.get((cell, stat, engine))
        if not cand_ns:
            return np.nan
        smaller = [m for m in cand_ns if m <= n]
        chosen = max(smaller) if smaller else min(cand_ns)
        return lut.get((cell, chosen, stat, engine, a), np.nan)

    null_sizes = pd.read_csv(RES / "size_calibration.csv")

    # ---- per-alpha power
    recs = []
    alphas_avail = sorted(set(ALPHAS) &
                          set(env["alpha_r"].unique()))
    for a in alphas_avail:
        d = df.copy()
        d["cv"] = [cv_for(r.cell, r.n, r.stat, r.engine, a)
                   for r in df.itertuples()]
        d = d[d["cv"].notna()]
        d["rej"] = d["stat_obs"] > d["cv"]
        g = (d.groupby(["cell", "family", "rho", "n", "stat", "engine"],
                       as_index=False)
             .agg(power=("rej", "mean"), n_reps=("rej", "size")))
        g["alpha"] = a
        recs.append(g)
    curve = pd.concat(recs, ignore_index=True)

    # ---- null size(alpha) from WP 2.2 (envelope decisions)
    sz = null_sizes[null_sizes["arm_class"].isin(
        ["clean", "contam_feasible"]) &
        (null_sizes["n_datasets"] >= 12)]
    sz_lu = {(r.cell, r.kind if False else None, r.n, r.stat, r.engine,
              round(float(r.alpha), 2)): r.size_envelope
             for r in sz.itertuples()}
    # kind-level granularity is unnecessary here; average over kinds
    sz_avg = (sz.groupby(["cell", "n", "stat", "engine", "alpha"],
                         as_index=False)
              .agg(size_05=("size_envelope", "mean")))
    sz_lu = {(r.cell, r.n, r.stat, r.engine, round(float(r.alpha), 2)):
             r.size_05 for r in sz_avg.itertuples()}

    def interp_at_size(cell, n, stat, engine, target):
        xs, ys = [], []
        for a in alphas_avail:
            s = sz_lu.get((cell, n, stat, engine, a), np.nan)
            p = curve_lu.get((cell, "mixture", None, n, stat, engine, a),
                             np.nan)
            xs.append(s)
            ys.append(p)
        xs, ys = np.array(xs, float), np.array(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if len(xs) < 3 or target < xs.min() or target > xs.max():
            return np.nan, float(np.nanmax(ys)) if len(ys) else np.nan
        order = np.argsort(xs)
        p_at = float(np.interp(target, xs[order], ys[order]))
        return p_at, float(np.interp(target, xs[order], ys[order]))

    # build curve lookup keyed for fast access (family=mixture primary;
    # mechanistic kept separately below)
    curve_lu = {(r.cell, r.family, None, r.n, r.stat, r.engine, r.alpha):
                r.power for r in curve.itertuples()}

    out = []
    mix = curve[curve["family"] == "mixture"]
    for key, g in mix.groupby(["cell", "n", "rho", "stat", "engine"]):
        cell, n, rho, stat, engine = key
        gg = g.sort_values("alpha")
        xs, ys = [], []
        for r in gg.itertuples():
            s = sz_lu.get((cell, n, stat, engine,
                           round(float(r.alpha), 2)), np.nan)
            xs.append(s)
            ys.append(r.power)
        xs, ys = np.array(xs, float), np.array(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        row = {"cell": cell, "family": "mixture", "n": n, "rho": rho,
               "stat": stat,
               "engine": engine, "mode": "matched_size"}
        for tgt in SIZE_TARGETS:
            if ok.sum() >= 3 and tgt <= xs[ok].max():
                order = np.argsort(xs[ok])
                row[f"power_at_size_{tgt:.2f}"] = float(np.interp(
                    tgt, xs[ok][order], ys[ok][order]))
            else:
                row[f"power_at_size_{tgt:.2f}"] = np.nan
        a05 = gg[gg["alpha"] == 0.05]
        row["size_at_0.05"] = float(sz_lu.get((cell, n, stat, engine,
                                               0.05), np.nan))
        row["power_at_0.05"] = float(a05["power"].iloc[0]) \
            if len(a05) else np.nan
        out.append(row)

    pc = pd.DataFrame(out).sort_values(
        ["cell", "n", "stat", "engine"])
    pc.to_csv(RES / "power_curves.csv", index=False)

    rt = (df.groupby(["cell", "n"])
          .agg(dt_mean=("dt_stats_s", "mean"),
               dt_median=("dt_stats_s", "median"),
               dt_max=("dt_stats_s", "max"))
          .reset_index())
    rt.to_csv(RES / "runtime_scaling.csv", index=False)
    print(f"power_curves.csv: {len(pc)} matched-size rows "
          f"(alphas {alphas_avail[0]}..{alphas_avail[-1]}); "
          f"runtime_scaling.csv: {len(rt)} rows")


if __name__ == "__main__":
    main()
