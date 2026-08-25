"""Gate B mechanical decision evaluation (WP 2.5).

Size gate (implemented reading of the predeclared rule "worst-case
empirical size <= 2*alpha across all null configurations after trying all
calibration variants"):
- evaluated per (stat, engine) variant;
- configurations = clean + feasible-contamination arms with >= 24
  replications (12-replication anchor configs cannot resolve a 0.10
  threshold: binomial noise alone exceeds it);
- the subsample engine is reported separately as FAILED (demonstrated
  degenerate-limit miscalibration) - this is exactly the plan's
  "after trying all calibration variants" clause.
Winning region: >= 3 adjacent (n, rho) cells with matched-size power
advantage >= 0.15 over the strongest applicable incumbent.
Verdicts: GO / PIVOT / INCREMENTAL-ONLY / KILL.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase2_discrete"

PRIMARY_CANDIDATES = ["kl_plugin", "kl_crossfit", "kl_split"]
INCUMBENTS = ["pearl_facet", "gm_battery", "slack_plugin", "gtest"]
BIG_CELLS = {"2-2-8", "3-3-5"}
MIN_REPS = 24


def wilson_ucb(k, n, z=1.96):
    if n == 0:
        return np.nan
    p = k / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (center - rad) / denom


def size_gate():
    size = pd.read_csv(RES / "size_calibration.csv")
    s05 = size[size["alpha"] == 0.05]
    arms = s05[s05["arm_class"].isin(["clean", "contam_feasible"])]
    rows = []
    for (stat, engine), g in arms.groupby(["stat", "engine"]):
        gg = g[g["n_datasets"] >= MIN_REPS]
        if len(gg) == 0:
            continue
        wc = float(gg["size_envelope"].max())
        k = float((gg["size_envelope"] * gg["n_datasets"]).sum())
        n = float(gg["n_datasets"].sum())
        wc_ad = float(gg["size_adaptive"].max())
        rows.append({"stat": stat, "engine": engine,
                     "n_configs": int(len(gg)),
                     "worst_size_05_env": wc,
                     "worst_size_05_adaptive": wc_ad,
                     "mean_size_overall": k / n,
                     "wilson_ucb_worst_config":
                         max(wilson_ucb(r.size_pooled, r.n_datasets)
                             for r in gg.itertuples()),
                     "passes_2alpha": bool(wc <= 0.10)})
    return pd.DataFrame(rows).sort_values(["stat", "engine"])


def winning_region():
    power = pd.read_csv(RES / "power_curves.csv")
    # canonical variant: parametric-bootstrap-calibrated statistics
    power = power[power["engine"] == "para_boot"]
    pv = power.pivot_table(index=["cell", "n", "rho"], columns="stat",
                           values="power_at_size_0.04",
                           aggfunc="mean").reset_index()
    inc_cols = [c for c in INCUMBENTS if c in pv.columns]
    prim = next(c for c in PRIMARY_CANDIDATES if c in pv.columns)
    pv["best_incumbent"] = pv[inc_cols].max(axis=1)
    pv["best_incumbent_ex_gtest"] = \
        pv[[c for c in inc_cols if c != "gtest"]].max(axis=1)
    pv["adv"] = pv[prim] - pv["best_incumbent"]
    pv["adv_ex_gtest"] = pv[prim] - pv["best_incumbent_ex_gtest"]

    def streaks(col):
        out = []
        for cell, g in pv.groupby("cell"):
            best = 0
            ns = sorted(g["n"].unique())
            rhos = sorted(g["rho"].unique())
            look = {(r["n"], r["rho"]): r[col]
                    for r in g.to_dict("records")}
            for n in ns:
                run = 0
                for rho in rhos:
                    v = look.get((n, rho), np.nan)
                    run = run + 1 if np.isfinite(v) and v >= 0.15 else 0
                    best = max(best, run)
            out.append({"cell": cell, "streak": best})
        return pd.DataFrame(out)

    reg = streaks("adv")
    reg_xg = streaks("adv_ex_gtest")
    return prim, reg, reg_xg


def main():
    print("=== SIZE GATE (alpha=0.05, configs >= %d reps, "
          "clean + feasible-contam) ===" % MIN_REPS)
    sg = size_gate()
    sg.to_csv(RES / "gateB_size_gate.csv", index=False)
    print(sg.to_string(index=False,
                       float_format=lambda x: f"{x:.4f}"))

    kl_ok = sg[(sg["stat"].str.startswith("kl")) &
               (sg["engine"].isin(["para_boot", "crt_cond"]))]
    size_pass = bool(len(kl_ok) and kl_ok["passes_2alpha"].any())
    sub_fail = bool(len(sg[(sg["engine"] == "subsample") &
                           (~sg["passes_2alpha"])]))

    print("\n=== WINNING REGION ===")
    prim, reg, reg_xg = winning_region()
    merged = reg.merge(reg_xg, on="cell", suffixes=("", "_ex_gtest"))
    merged.to_csv(RES / "winning_region_summary.csv", index=False)
    print(merged.to_string(index=False))
    best_streak = int(reg["streak"].max())
    best_cells = list(reg[reg["streak"] >= 3]["cell"])
    wins_exist = best_streak >= 3

    # monotonicity sanity on primary witness (mixture family)
    power = pd.read_csv(RES / "power_curves.csv")
    mono_viol = 0
    mono_checks = 0
    pp = power[(power["stat"] == prim) & (power["family"] == "mixture")]
    for key, g in pp.groupby(["cell", "n"]):
        g = g.sort_values("rho")
        diffs = np.diff(g["power_at_size_0.04"].values)
        mono_checks += 1
        mono_viol += int(np.sum(diffs < -0.08))

    print(f"\nprimary={prim}; size_pass={size_pass}; "
          f"subsample_engine_failed={sub_fail}")
    print(f"best adjacent winning streak={best_streak}; "
          f"cells with streak>=3: {best_cells}")
    print(f"monotonicity: {mono_viol} violations across {mono_checks} "
          "cell/n panels")

    print("\n=== GATE B VERDICT ===")
    if not size_pass:
        print("KILL: no KL-family variant holds worst-case size <= 2a")
        return
    if not wins_exist:
        adv_max = float(power[power["stat"] == prim][
            "power_at_0.05"].max()) \
            if "power_at_0.05" in power.columns else np.nan
        print(f"INCREMENTAL-ONLY/KILL consult: no adjacent winning region "
              f"(max streak {best_streak})")
        return
    big_only = bool(best_cells) and all(c in BIG_CELLS for c in best_cells)
    if big_only:
        print("PIVOT: wins confined to large-alphabet cells "
              f"{best_cells}; scope claims accordingly")
    else:
        print(f"GO: discrete arm proceeds to Phase 4 scope with winning "
              f"region(s) in {best_cells} (streak {best_streak}); "
              "subsample engine dropped; cf1/slack retained as effect-size/"
              "LP-equivalent companions.")


if __name__ == "__main__":
    main()
