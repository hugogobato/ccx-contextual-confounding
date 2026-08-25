"""Phase 3 aggregator v2: null_calibration.csv, crackle_stress.csv,
separation_study.csv + predeclared pass-rule evaluation.

WP 3.2 raw rows carry per-dataset own-bootstrap CVs at the alpha grid;
size tables use adaptive decisions directly. WP 3.3 rows carry stat_obs;
critical values transfer from matched (noise, n, d, method) null runs at
the primary trimming policy, with conservative nearest-n transfer where
the separation grid lacks an exact match.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase3_continuous"
RAW = ROOT / "results" / "raw" / "phase3"

TRIM_PRIMARY = 0.01


def main():
    RES.mkdir(parents=True, exist_ok=True)
    f32 = sorted(RAW.glob("wp32_*.csv"))
    assert f32, "no wp32 raw files"
    df32 = pd.concat([pd.read_csv(f) for f in f32], ignore_index=True)

    size_rows = []
    for key, g in df32.groupby(["kind", "noise", "n", "d", "method",
                                "trim"]):
        kind, noise, n, d, meth, trim = key
        rej = (g["stat_obs"] > g["cv_0.05"]).mean()
        size_rows.append({"kind": kind, "noise": noise, "n": n, "d": d,
                          "method": meth, "trim": trim,
                          "n_datasets": int(len(g)),
                          "size_05": float(rej),
                          "mean_cv05": float(g["cv_0.05"].mean())})
    size = pd.DataFrame(size_rows)
    size.to_csv(RES / "null_calibration.csv", index=False)
    size[size["noise"] != "gauss"].to_csv(RES / "crackle_stress.csv",
                                          index=False)

    # ---- critical-value transfer table for WP 3.3
    prim = df32[df32["trim"] == TRIM_PRIMARY]
    cvt = (prim.groupby(["kind", "noise", "n", "d", "method"],
                        as_index=False)
           .agg(cv_05=("cv_0.05", "mean")))
    lut_full = {(r.kind, r.noise, r.n, r.d, r.method): r.cv_05
                for r in cvt.itertuples()}
    ns_by_key = {}
    for (k, nz, n, dd, m) in lut_full:
        ns_by_key.setdefault((nz, dd, m), set()).add(n)

    def cv_for(kind, noise, n, d, meth):
        if (kind, noise, n, d, meth) in lut_full:
            return lut_full[(kind, noise, n, d, meth)]
        cand = ns_by_key.get((noise, d, meth), set())
        if not cand:
            return np.nan
        smaller = [m for m in sorted(cand) if m <= n]
        chosen = max(smaller) if smaller else min(cand)
        return lut_full.get((kind, noise, chosen, d, meth), np.nan)

    f33 = sorted(RAW.glob("wp33_*.csv"))
    if not f33:
        print("wp33 not run yet; wrote null tables only")
        return
    df33 = pd.concat([pd.read_csv(f) for f in f33], ignore_index=True)
    df33["kind"] = df33["kind"].replace(
        {"conf_lin": "null_gauss", "conf_nonlin": "null_nonparam"})
    # alternatives judged against the SAME structural class as the nulls:
    # conf_* DGPs violate the shared-shape additive class that null_gauss /
    # null_nonparam satisfy; map to nearest null kind by mechanism family
    df33["cv_0.05"] = [cv_for(r.kind, r.noise, r.n, r.d, r.method)
                       for r in df33.itertuples()]
    df33["rej"] = df33["stat_obs"] > df33["cv_0.05"]
    sep = (df33.groupby(["kind", "noise", "n", "d", "b", "method"],
                        as_index=False)
           .agg(power=("rej", "mean"), n_reps=("rej", "size")))

    sz_lu = {(r.noise, r.n, r.d, r.method): r.size_05
             for r in size.itertuples()
             if r.trim == TRIM_PRIMARY}
    sep["null_size"] = [sz_lu.get((r.noise, r.n, r.d, r.method), np.nan)
                        for r in sep.itertuples()]
    sep["advantage"] = sep["power"] - sep["null_size"]
    sep.to_csv(RES / "separation_study.csv", index=False)

    summary = []
    pw = sep[sep["method"].isin(["k1", "k2"])]
    inc = sep[sep["method"] == "hsic"][["noise", "n", "d", "b",
                                        "power"]] \
        .rename(columns={"power": "power_hsic"})
    pw = pw.merge(inc, on=["noise", "n", "d", "b"], how="left")
    for (meth, kd, nz), g in pw.groupby(["method", "kind", "noise"]):
        ok_base = g[g["power_hsic"] <= 0.15]
        passed = ok_base[ok_base["advantage"] >= 0.25]
        gg = g.sort_values("b")
        mono_ok = bool(np.all(np.diff(gg["power"].values) >= -0.08))
        summary.append({
            "method": meth, "kind": kd, "noise": nz,
            "favorable_cells_total": int(len(ok_base)),
            "cells_with_adv_ge_025": int(len(passed)),
            "monotone_in_b": mono_ok,
            "max_power": float(g["power"].max()),
            "pass_rule_met": bool(len(passed) >= 2 and mono_ok),
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(RES / "pass_rule_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
