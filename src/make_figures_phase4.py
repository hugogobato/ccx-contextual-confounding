"""Phase 4 figures (regenerable from results/phase4_grid CSVs).

Outputs to figures/phase4/:
- cont_power_panels.png       power vs b, k1/k2/hsic, per (kind, noise),
                              one column per n (d=2 panel)
- discrete_power_panels.png   power vs rho, best witness variant vs best
                              LP variant, per (cell, tail) at n=2000
- adversarial_fpr.png         FPR per adversarial config and method
- scaling_phase4.png          wall seconds vs n (continuous) and vs cell
                              (discrete)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase4_grid"
FIG = ROOT / "figures" / "phase4"

P4 = None


def fig_continuous(sep):
    cg = (P4 or __import__("json").loads(
        (ROOT / "configs" / "phase4.json").read_text()))["continuous_grid"]
    kinds = cg["alt_kinds"]
    noises = cg["noise"]
    fig, axes = plt.subplots(len(kinds) * len(noises), len(cg["n_grid"]),
                             figsize=(4 * len(cg["n_grid"]),
                                      3 * len(kinds) * len(noises)),
                             squeeze=False, sharey=True)
    for i, kd in enumerate(kinds):
        for j, nz in enumerate(noises):
            for k, n in enumerate(cg["n_grid"]):
                ax = axes[i * len(noises) + j][k]
                for meth, ls in (("k1", "-"), ("k2", "--"), ("hsic", ":")):
                    sub = sep[(sep["method"] == meth) & (sep["kind"] == kd) &
                              (sep["noise"] == nz) & (sep["n"] == n) &
                              (sep["d"] == 2)].sort_values("b")
                    if len(sub):
                        ax.plot(sub["b"], sub["power"], ls,
                                marker="o", ms=3,
                                label=f"{meth} (size {sub['null_size'].iloc[0]:.2f})"
                                if len(sub) else meth)
                ax.set_title(f"{kd} / {nz} / n={n}")
                ax.set_xlabel("b"); ax.set_ylabel("power")
                ax.set_ylim(-0.02, 1.05)
                if i == 0 and j == 0 and k == 0:
                    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "cont_power_panels.png", dpi=150)
    plt.close(fig)


def _best(pw, cell, tail, n, rho, pairs):
    best = np.nan
    for s, e in pairs:
        r = pw[(pw["cell"] == cell) & (pw["tail"] == tail) &
               (pw["n"] == n) & (pw["rho"] == rho) &
               (pw["stat"] == s) & (pw["engine"] == e)]
        if len(r) and np.isfinite(r["power"].iloc[0]):
            best = max(best, float(r["power"].iloc[0])) \
                if np.isfinite(best) else float(r["power"].iloc[0])
    return best


def fig_discrete(pw):
    p4 = P4 or __import__("json").loads(
        (ROOT / "configs" / "phase4.json").read_text())
    dgx = p4["discrete_grid"]
    w_pairs = [(s, e) for s in ("kl_plugin", "kl_split", "kl_crossfit")
               for e in ("para_boot", "crt_cond")]
    b_pairs = [(s, e) for s in ("cf1_margin", "slack_plugin")
               for e in ("para_boot", "crt_cond")]
    fig, axes = plt.subplots(len(dgx["cells"]), len(dgx["tails"]),
                             figsize=(9, 3 * len(dgx["cells"])),
                             squeeze=False, sharey=True)
    for i, cell in enumerate(dgx["cells"]):
        cid = "-".join(map(str, cell))
        for j, tail in enumerate(dgx["tails"]):
            ax = axes[i][j]
            rhos, pw_v, pw_b = [], [], []
            for rho in dgx["rho_grid"]:
                rhos.append(rho)
                pw_v.append(_best(pw, cid, tail, 2000, rho, w_pairs))
                pw_b.append(_best(pw, cid, tail, 2000, rho, b_pairs))
            ax.plot(rhos, pw_v, "-o", label="KL witness (best size-ok)")
            ax.plot(rhos, pw_b, "--s", label="calibrated LP (best)")
            ax.set_title(f"{cid} / tail={tail} / n=2000")
            ax.set_xlabel("rho"); ax.set_ylabel("power")
            ax.set_ylim(-0.02, 1.05)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "discrete_power_panels.png", dpi=150)
    plt.close(fig)


def fig_adversarial():
    d1 = RES / "discrete_adversarial.csv"
    d2 = RES / "continuous_adversarial.csv"
    rows = []
    if d1.exists():
        t = pd.read_csv(d1)
        for r in t.itertuples():
            rows.append({"config": r.adv_id, "method": r.stat,
                         "fpr": r.fpr})
    if d2.exists():
        t = pd.read_csv(d2)
        for r in t.itertuples():
            rows.append({"config": r.adv_id, "method": r.method,
                         "fpr": r.fpr})
    if not rows:
        return
    df = pd.DataFrame(rows)
    keep = df[df["method"].isin(["kl_plugin", "k1", "k2", "cf1_margin"])]
    fig, ax = plt.subplots(figsize=(10, 4))
    configs = keep["config"].unique()
    width = 0.8 / keep["method"].nunique()
    for i, meth in enumerate(sorted(keep["method"].unique())):
        vals = [keep[(keep["config"] == c) &
                     (keep["method"] == meth)]["fpr"].max() * 0 +
                keep[(keep["config"] == c) &
                     (keep["method"] == meth)]["fpr"].mean()
                for c in configs]
        ax.bar(np.arange(len(configs)) + i * width, vals, width,
               label=meth)
    ax.axhline(0.10, color="red", ls="--", lw=1, label="Gate D limit 0.10")
    ax.axhline(0.05, color="gray", ls=":", lw=1, label="alpha 0.05")
    ax.set_xticks(np.arange(len(configs)) + width)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("empirical FPR (alpha=0.05)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "adversarial_fpr.png", dpi=150)
    plt.close(fig)


def fig_scaling(sc):
    c = sc[sc["arm"] == "continuous"]
    d = sc[sc["arm"] == "discrete"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for meth, g in c.groupby("method"):
        g = g.groupby("n")["seconds"].median().reset_index()
        axes[0].plot(g["n"], g["seconds"], "-o", label=meth)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("n"); axes[0].set_ylabel("median seconds")
    axes[0].legend(fontsize=8); axes[0].set_title("continuous")
    for meth, g in d.groupby("method"):
        g = g.groupby(["cell", "n"])["seconds"].median().reset_index()
        for cell, gg in g.groupby("cell"):
            axes[1].plot(gg["n"], gg["seconds"], "-o", alpha=0.7,
                         label=f"{meth} {cell}" if meth != "stats_all"
                         else None)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("n"); axes[1].set_ylabel("median seconds")
    axes[1].legend(fontsize=7); axes[1].set_title("discrete")
    fig.tight_layout()
    fig.savefig(FIG / "scaling_phase4.png", dpi=150)
    plt.close(fig)


def main():
    global P4
    FIG.mkdir(parents=True, exist_ok=True)
    P4 = __import__("json").loads(
        (ROOT / "configs" / "phase4.json").read_text())
    sep_f = RES / "continuous_separation.csv"
    pw_f = RES / "discrete_power_curves.csv"
    sc_f = RES / "scaling_phase4.csv"
    if sep_f.exists():
        fig_continuous(pd.read_csv(sep_f))
        print("cont_power_panels done")
    if pw_f.exists():
        fig_discrete(pd.read_csv(pw_f))
        print("discrete_power_panels done")
    fig_adversarial()
    print("adversarial_fpr done")
    if sc_f.exists():
        fig_scaling(pd.read_csv(sc_f))
        print("scaling_phase4 done")


if __name__ == "__main__":
    main()
