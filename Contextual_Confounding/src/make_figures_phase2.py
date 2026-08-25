"""Phase 2 figures (regenerable from results CSVs only).

Outputs under figures/phase2/:
- fig_size_vs_n.png            : empirical size vs n, alpha=0.05, pooled mode
- fig_power_binary.png         : binary-cell power vs rho at matched sizes
- fig_advantage_heatmap.png    : KL-witness advantage vs strongest incumbent
- fig_lp_walltime_map.png      : WP 2.4 runtime pain map
- fig_contamination_detect.png : corrupted-data detection rates
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase2_discrete"
FIG = ROOT / "figures" / "phase2"

INCUMBENTS = ["pearl_facet", "gm_battery", "slack_plugin", "gtest"]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    size = pd.read_csv(RES / "size_calibration.csv")
    power = pd.read_csv(RES / "power_curves.csv")
    wall = pd.read_csv(RES / "lp_walltime_map.csv")
    try:
        det = pd.read_csv(RES / "contamination_detection.csv")
    except FileNotFoundError:
        det = None

    # ---------------- size vs n
    cl = size[(size["arm_class"] == "clean") & (size["alpha"] == 0.05)]
    stats_order = ["kl_plugin", "kl_split", "kl_crossfit", "cf1_margin",
                   "slack_plugin", "pearl_facet", "gm_battery", "gtest"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True,
                             sharex=True)
    for ax, st in zip(axes.ravel(), stats_order):
        d = cl[(cl["stat"] == st) & (cl["engine"].isin(
            ["para_boot", "none"]))]
        piv = d.groupby(["cell", "n"])["size_pooled"].mean().reset_index()
        for cell, g in piv.groupby("cell"):
            ax.plot(g["n"], g["size_pooled"], marker="o", ms=3, lw=1,
                    label=f"cell {cell}")
        ax.axhline(0.05, color="k", lw=0.8, ls="--")
        ax.axhspan(0, 0.10, color="red", alpha=0.06)
        ax.set_xscale("log")
        ax.set_title(st, fontsize=9)
        ax.set_ylim(0, 0.35)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("WP 2.2 empirical size vs n (alpha=0.05, pooled CVs; "
                 "red band = <=2alpha)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_size_vs_n.png", dpi=150)
    plt.close(fig)

    # ---------------- detection-boundary comparison
    pb = power[power["engine"] == "para_boot"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    panels = [("2-2-3", 250), ("3-3-3", 500), ("3-3-5", 500)]
    for ax, (cell, n) in zip(axes, panels):
        d = pb[(pb.cell == cell) & (pb.n == n)]
        for st, g in d.groupby("stat"):
            g = g.sort_values("rho")
            ax.plot(g["rho"], g["power_at_0.05"], marker="o", ms=3,
                    lw=1.2, label=st)
        ax.set_title(f"{cell}, n={n}", fontsize=9)
        ax.set_xlabel("conflict mass rho")
        ax.axhline(0.05, color="k", lw=0.7, ls=":")
    axes[0].set_ylabel("power at nominal alpha=0.05")
    axes[0].legend(fontsize=6)
    fig.suptitle("Detection-boundary power (mixture family)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_power_boundary.png", dpi=150)
    plt.close(fig)

    # ---------------- binary power curves
    b = power[(power["family"] == "mixture") & (power["cell"] == "2-2-2")]
    if len(b):
        ns = sorted(b["n"].unique())
        fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.8),
                                 sharey=True)
        if len(ns) == 1:
            axes = [axes]
        for ax, nv in zip(np.atleast_1d(axes), ns):
            bn = b[b["n"] == nv]
            for st, g in bn.groupby("stat"):
                g = g.sort_values("rho")
                style = "-" if st.startswith(("kl", "cf1")) else "--"
                ax.plot(g["rho"], g["power_at_0.05"], style,
                        marker="o", ms=3, lw=1.2, label=st)
            ax.set_title(f"binary IV, n={nv}", fontsize=9)
            ax.set_xlabel("confounded mass rho")
            ax.axhline(0.04, color="k", lw=0.7, ls=":")
        axes[0].set_ylabel("power at nominal alpha=0.05")
        axes[0].legend(fontsize=6)
        fig.suptitle("WP 2.3 power, mixture family (binary cell)")
        fig.tight_layout()
        fig.savefig(FIG / "fig_power_binary.png", dpi=150)
        plt.close(fig)

    # ---------------- advantage heatmap: primary witness vs best incumbent
    pw = power[power["mode"] == "pooled"].copy()
    pv = pw.pivot_table(index=["cell", "family", "n", "rho"],
                        columns="stat",
                        values="power_at_size_0.04").reset_index()
    inc_cols = [c for c in INCUMBENTS if c in pv.columns]
    if "kl_crossfit" in pv.columns and inc_cols:
        pv["best_incumbent"] = pv[inc_cols].max(axis=1)
        prim = "kl_crossfit" if "kl_crossfit" in pv.columns else "kl_plugin"
        pv["advantage"] = pv[prim] - pv["best_incumbent"]
        cells = sorted(pv["cell"].unique())
        ns = sorted(pv["n"].unique())
        fig, axes = plt.subplots(len(cells), len(ns),
                                 figsize=(2.6 * len(ns), 2.0 * len(cells)),
                                 squeeze=False)
        for i, c in enumerate(cells):
            for j, nv in enumerate(ns):
                d = pv[(pv["cell"] == c) & (pv["n"] == nv)]
                grid = d.pivot_table(index="rho", values="advantage")
                im = axes[i][j].imshow(grid.values.reshape(-1, 1),
                                       aspect="auto", cmap="RdBu_r",
                                       vmin=-0.2, vmax=0.2)
                axes[i][j].set_yticks(range(len(grid)))
                axes[i][j].set_yticklabels([f"{r:.1f}" for r in
                                            grid.index], fontsize=5)
                axes[i][j].set_xticks([])
                axes[i][j].set_title(f"{c} n={nv}", fontsize=6)
        fig.colorbar(im, ax=axes, shrink=0.6, label="power advantage")
        fig.suptitle(f"KL-crossfit advantage over best incumbent "
                     f"(power@size0.04)", y=0.995)
        fig.savefig(FIG / "fig_advantage_heatmap.png", dpi=150)
        plt.close(fig)

    # ---------------- LP wall-time map
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lp = wall[wall["method"].isin(["cf1_lp", "slack_lp",
                                   "inflation_order2", "null_fit_em",
                                   "kl_plugin_single"])]
    for m, g in lp.groupby("method"):
        gg = g.groupby("Q").agg(sec=("seconds", "median"),
                                secmax=("seconds", "max")).reset_index()
        axes[0].plot(gg["Q"], gg["sec"], marker="o", ms=4, label=m)
        axes[1].plot(gg["Q"], np.maximum(gg["secmax"], 1e-4), marker="o",
                     ms=4, label=m)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("coupling dimension Q (= kx^kz * ky^kx)")
        ax.set_ylabel("seconds (log)")
        ax.legend(fontsize=7)
    axes[0].set_title("median per-call time")
    axes[1].set_title("worst observed per-call time")
    fig.suptitle("WP 2.4 pain map: LP-family vs witness-family scaling",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_lp_walltime_map.png", dpi=150)
    plt.close(fig)

    # ---------------- contamination detection
    if det is not None and len(det):
        d05 = det[det["alpha"] == 0.05]
        stats_sel = ["kl_plugin", "kl_crossfit", "gm_battery",
                     "pearl_facet", "gtest"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for st in stats_sel:
            if st not in set(d05["stat"]):
                continue
            g = (d05[d05["stat"] == st]
                 .groupby("n")["power_pooled"].mean())
            ax.plot(g.index, g.values, marker="o", ms=4, label=st)
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("detection rate (infeasible contaminated nulls)")
        ax.legend(fontsize=7)
        ax.set_title("Data-corruption detection (eps=0.05, alpha=0.05)")
        fig.tight_layout()
        fig.savefig(FIG / "fig_contamination_detect.png", dpi=150)
        plt.close(fig)

    print("figures written to", FIG)


if __name__ == "__main__":
    main()
