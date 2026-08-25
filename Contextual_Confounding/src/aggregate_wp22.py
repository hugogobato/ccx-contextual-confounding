"""WP 2.2 aggregator: pooled critical values, size tables, variant selection.

Outputs (results/phase2_discrete/):
- null_critical_values.csv : pooled CV per (cell, n, stat, engine, alpha),
  averaged over clean-arm bootstrap-enabled datasets (documented
  aggregation; draws never leave the workers).
- size_calibration.csv     : empirical sizes per config x stat x engine x
  mode (adaptive = own-draw CVs, pooled = ensemble CVs), clean and
  feasible-contamination arms; plus detection rates for infeasible-contam.
- selected_variants.csv    : predeclared selection rule output.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase2_discrete"
RAW = ROOT / "results" / "raw" / "phase2"

BINDING = [0.05, 0.10]


def main():
    files = sorted(RAW.glob("wp22_*.csv"))
    assert files, "no wp22 raw files found"
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["src_file"] = f.name
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["arm_class"] = np.where(
        df["arm"] == "clean", "clean",
        np.where(df["feas_contam"].astype(bool), "contam_feasible",
                 "contam_infeasible"))
    alpha_cols = [c for c in df.columns if c.startswith("cv_")]
    alphas = [float(c[3:]) for c in alpha_cols]

    # ---------------- pooled critical values (clean arm, boot-enabled)
    boot = df[(df["arm_class"] == "clean") & (df["B"] > 0)]
    pooled = (boot.groupby(["cell", "n", "stat", "engine"],
                           as_index=False)[alpha_cols].mean())
    pv = pooled.melt(id_vars=["cell", "n", "stat", "engine"],
                     var_name="alpha_s", value_name="cv_pooled")
    pv["alpha"] = pv["alpha_s"].str[3:].astype(float)
    pv.drop(columns="alpha_s").to_csv(RES / "null_critical_values.csv",
                                      index=False)

    # ---- per-kind CVs + envelope (max over kinds): conditional
    # calibration guard against law-heterogeneous null scales
    boot_kinds = df[(df["arm_class"] == "clean") & (df["B"] > 0)]
    pk = (boot_kinds.groupby(["cell", "kind", "n", "stat", "engine"],
                             as_index=False)[alpha_cols].mean())
    env = (pk.groupby(["cell", "n", "stat", "engine"],
                      as_index=False)[alpha_cols].max())
    ev = env.melt(id_vars=["cell", "n", "stat", "engine"],
                  var_name="alpha_s", value_name="cv_env")
    ev["alpha"] = ev["alpha_s"].str[3:].astype(float)
    ev.drop(columns="alpha_s").to_csv(
        RES / "null_critical_values_envelope.csv", index=False)
    env_lookup = {(r.cell, r.n, r.stat, r.engine, round(r.alpha, 2)):
                  r.cv_env for r in ev.itertuples()}
    cv_lookup = {(r.cell, r.n, r.stat, r.engine, round(r.alpha, 2)):
                 r.cv_pooled for r in pv.itertuples()}

    # ---------------- rejection indicators
    # itertuples mangles dotted column names; use a renamed frame
    df_r = df.rename(columns={c: f"cva{c[3:].replace('.', '_')}"
                              for c in alpha_cols})

    def rej_rows(sub):
        sub = sub.rename(columns={c: f"cva{c[3:].replace('.', '_')}"
                                  for c in alpha_cols})
        out = []
        for r in sub.itertuples():
            for c, a in zip(alpha_cols, alphas):
                key = (r.cell, r.n, r.stat, r.engine, round(a, 2))
                own = getattr(r, f"cva{c[3:].replace('.', '_')}")
                pcv = cv_lookup.get(key, np.nan)
                ecv = env_lookup.get(key, np.nan)
                out.append((r.cell, r.kind, r.n, r.arm_class, r.stat,
                            r.engine, a,
                            bool(r.stat_obs > own) if np.isfinite(own)
                            else np.nan,
                            bool(r.stat_obs > pcv) if np.isfinite(pcv)
                            else np.nan,
                            bool(r.stat_obs > ecv) if np.isfinite(ecv)
                            else np.nan))
        return out

    recs = rej_rows(df)
    rej = pd.DataFrame(recs, columns=["cell", "kind", "n", "arm_class",
                                      "stat", "engine", "alpha",
                                      "rej_adaptive", "rej_pooled",
                                      "rej_envelope"])

    # ---------------- size table: clean + contam_feasible are SIZE arms
    size_arms = rej[rej["arm_class"].isin(["clean", "contam_feasible"])]
    det_arms = rej[rej["arm_class"] == "contam_infeasible"]

    def agg(g):
        return pd.Series({
            "n_datasets": int(len(g)),
            "size_adaptive": float(g["rej_adaptive"].mean()),
            "size_pooled": float(g["rej_pooled"].mean()),
            "size_envelope": float(g["rej_envelope"].mean()),
        })

    size = (size_arms.groupby(["cell", "kind", "n", "arm_class", "stat",
                               "engine", "alpha"])
            .apply(agg, include_groups=False).reset_index())
    det = (det_arms.groupby(["cell", "kind", "n", "stat", "engine", "alpha"])
           .apply(agg, include_groups=False).reset_index()
           .rename(columns={"size_adaptive": "power_adaptive",
                            "size_pooled": "power_pooled",
                            "size_envelope": "power_envelope"}))

    size.to_csv(RES / "size_calibration.csv", index=False)
    det.to_csv(RES / "contamination_detection.csv", index=False)

    # ---------------- worst-case summaries + variant selection
    clean = size[size["arm_class"] == "clean"]
    rows_sel = []
    print("\n==== WORST-CASE SIZE (clean nulls, binding alphas) ====")
    for (stat, engine), g in clean.groupby(["stat", "engine"]):
        row = {"stat": stat, "engine": engine}
        ok_all = True
        for mode in ("adaptive", "pooled"):
            for a in BINDING:
                ga = g[g["alpha"] == a]
                wc = float(ga[f"size_{mode}"].max())
                med = float(ga[f"size_{mode}"].median())
                mn = float(ga[f"size_{mode}"].min())
                row[f"wc_{mode}_a{a:.2f}"] = wc
                row[f"med_{mode}_a{a:.2f}"] = med
                row[f"min_{mode}_a{a:.2f}"] = mn
                if not (wc <= 2 * a + 1e-12 and mn >= 0.5 * a - 1e-12):
                    ok_all = False if mode == "pooled" else ok_all
        row["eligible_pooled"] = bool(all(
            row[f"wc_pooled_a{a:.2f}"] <= 2 * a + 1e-12 and
            row[f"min_pooled_a{a:.2f}"] >= 0.5 * a - 1e-12
            for a in BINDING))
        row["eligible_adaptive"] = bool(all(
            row[f"wc_adaptive_a{a:.2f}"] <= 2 * a + 1e-12 and
            row[f"min_adaptive_a{a:.2f}"] >= 0.5 * a - 1e-12
            for a in BINDING))
        rows_sel.append(row)
    sel = pd.DataFrame(rows_sel).sort_values(["stat", "engine"])
    sel.to_csv(RES / "selected_variants.csv", index=False)

    cols_show = ["stat", "engine", "wc_pooled_a0.05", "wc_adaptive_a0.05",
                 "wc_pooled_a0.10", "eligible_pooled", "eligible_adaptive"]
    print(sel[[c for c in cols_show if c in sel.columns]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # feasibility classification summary for contamination arms
    cf = df[df["arm_class"] != "clean"].groupby(
        ["cell", "arm_class"]).size().rename("n_rows").reset_index()
    cf.to_csv(RES / "contamination_classification_summary.csv", index=False)
    print("\nwrote: null_critical_values.csv, size_calibration.csv, "
          "selected_variants.csv, contamination_detection.csv")


if __name__ == "__main__":
    main()
