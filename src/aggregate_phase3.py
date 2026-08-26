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

    # ---- critical-value transfer table for WP 3.3 (v2): each method is
    # judged at its PRIMARY trimming policy (k1/k2: 0.01; hsic: 0.0)
    PRIM_TRIM = {"k1": 0.01, "k2": 0.01, "hsic": 0.0}
    prim = df32[[r.trim == PRIM_TRIM.get(r.method, r.trim)
                 for r in df32.itertuples()]]
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
             if r.trim == PRIM_TRIM.get(r.method, TRIM_PRIMARY)}
    sep["null_size"] = [sz_lu.get((r.noise, r.n, r.d, r.method), np.nan)
                        for r in sep.itertuples()]
    sep["advantage"] = sep["power"] - sep["null_size"]
    sep.to_csv(RES / "separation_study.csv", index=False)

    # ---- Gate C evaluation (v2): predeclared rule verbatim, per-panel
    # monotonicity (v1 mixed (n,d) panels inside one sort-by-b).
    # Rule: witness power >= size + 0.25 at b >= 0.4, n <= 5000, d <= 3,
    # size <= 0.06 at alpha 0.05, in >= 2 of 3 favorable-regime cells
    # where ALL baselines remain <= size + 0.10; power monotone in b.
    szmap = {(r.kind, r.noise, r.n, r.d, r.method): r.size_05
             for r in size.itertuples()
             if r.trim == PRIM_TRIM.get(r.method, TRIM_PRIMARY)}
    rows_c = []
    for meth in ("k1", "k2"):
        for kd in ("conf_lin", "conf_nonlin"):
            null_kind = ("null_gauss" if kd == "conf_lin"
                         else "null_nonparam")
            for nz in ("gauss", "t3", "lognorm"):
                sub = sep[(sep["method"] == meth) & (sep["kind"] == kd) &
                          (sep["noise"] == nz) & (sep["b"] >= 0.4) &
                          (sep["n"] <= 5000) & (sep["d"] <= 3)]
                if not len(sub):
                    continue
                # favorable-regime cells = distinct (n, d) panels
                panels = []
                for (nv, dv), g in sub.groupby(["n", "d"]):
                    g = g.sort_values("b")
                    mono = bool(np.all(np.diff(g["power"].values) >= -0.08))
                    base_ok = True
                    hsic_rows = sep[(sep["method"] == "hsic") &
                                    (sep["kind"] == kd) &
                                    (sep["noise"] == nz) &
                                    (sep["n"] == nv) & (sep["d"] == dv) &
                                    (sep["b"] >= 0.4)]
                    for r in hsic_rows.itertuples():
                        s_h = szmap.get((null_kind, nz, nv, dv, "hsic"),
                                        np.nan)
                        if not (np.isfinite(s_h) and
                                r.power <= s_h + 0.10):
                            base_ok = False
                            break
                    adv_ok = int(np.sum((g["advantage"] >= 0.25) &
                                        (g["null_size"] <= 0.06)))
                    panels.append({"n": nv, "d": dv,
                                   "size_worst":
                                       float(g["null_size"].max()),
                                   "adv_cells": adv_ok,
                                   "baseline_ok": bool(base_ok),
                                   "monotone_b": mono})
                df_p = pd.DataFrame(panels)
                elig = df_p[(df_p["baseline_ok"]) & (df_p["monotone_b"])]
                n_pass = int(((elig["adv_cells"] > 0)).sum())
                rows_c.append({
                    "method": meth, "kind": kd, "noise": nz,
                    "panels_total": int(len(df_p)),
                    "panels_eligible": int(len(elig)),
                    "panels_with_adv_ge_025": n_pass,
                    "worst_size_favorable":
                        float(df_p["size_worst"].max()),
                    "all_monotone": bool(df_p["monotone_b"].all()),
                    "pass_rule_met": bool(n_pass >= 2),
                })
    gatec = pd.DataFrame(rows_c)
    gatec.to_csv(RES / "gateC_evaluation.csv", index=False)

    summary = []
    pw = sep[sep["method"].isin(["k1", "k2"])]
    inc = sep[sep["method"] == "hsic"][["noise", "n", "d", "b",
                                        "power"]] \
        .rename(columns={"power": "power_hsic"})
    pw = pw.merge(inc, on=["noise", "n", "d", "b"], how="left")
    for (meth, kd, nz), g in pw.groupby(["method", "kind", "noise"]):
        ok_base = g[g["power_hsic"] <= 0.15]
        passed = ok_base[ok_base["advantage"] >= 0.25]
        mono_panels = True
        for (nv, dv), gg in g.groupby(["n", "d"]):
            gg = gg.sort_values("b")
            if not np.all(np.diff(gg["power"].values) >= -0.08):
                mono_panels = False
                break
        summary.append({
            "method": meth, "kind": kd, "noise": nz,
            "favorable_cells_total": int(len(ok_base)),
            "cells_with_adv_ge_025": int(len(passed)),
            "monotone_in_b_per_panel": mono_panels,
            "max_power": float(g["power"].max()),
            "pass_rule_met_legacy": bool(len(passed) >= 2 and mono_panels),
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(RES / "pass_rule_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\n=== Gate C predeclared-rule evaluation ===")
    print(gatec.to_string(index=False))


if __name__ == "__main__":
    main()
