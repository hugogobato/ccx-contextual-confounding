"""Phase 1 figures (regenerable from results CSVs only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase1_enumeration"
FIG = ROOT / "figures" / "phase1"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "t1_dictionary.csv", low_memory=False)
    hier = pd.read_csv(RES / "hierarchy_placement.csv")
    red = pd.read_csv(RES / "witness_lp_redundancy.csv")

    # ---- fig 1: dictionary agreement summary
    fig, ax = plt.subplots(figsize=(6, 3.4))
    cats = ["agree", "disagree", "borderline"]
    vals = [int(df["agree_c1a"].sum()),
            int((~df["agree_c1a"].astype(bool)).sum()),
            int(df["borderline_band"].sum())]
    ax.bar(cats, vals, color=["#3b7dd8", "#d83b3b", "#d8a53b"])
    ax.set_yscale("symlog")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"C1a dictionary agreement (tol 1e-9): "
                 f"{100*vals[0]/max(vals[0]+vals[1],1):.4f}%")
    ax.set_ylabel("instances (symlog)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_c1a_agreement.png", dpi=150)
    plt.close(fig)

    # ---- fig 2: witness vs LP slack scatters (subsample for speed)
    sub = df.sample(n=min(60000, len(df)), random_state=0)
    ctx = sub[~sub["lp_feasible"].astype(bool) & ~sub["borderline_band"].astype(bool)]
    wit_cols = [("cf1_soft", "cf1_soft (fraction t*)"),
                ("kl_contextuality", "KL contextuality"),
                ("degree_signed", "degree signed (CbD)"),
                ("slack_l1", "LP slack (L1)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False)
    for ax, (col, lab) in zip(axes.ravel(), wit_cols):
        if col == "slack_l1":
            ax.scatter(sub["slack_l1"], np.zeros_like(sub["slack_l1"]) +
                       sub["source"].eq("uniform"), s=2, alpha=0.05)
            ax.set_xlabel(lab); ax.set_yticks([])
            continue
        d = sub[[col, "slack_l1"]].dropna()
        rho = spearmanr(d[col], d["slack_l1"]).statistic \
            if d[col].nunique() > 1 else float("nan")
        ax.scatter(d[col], d["slack_l1"], s=2, alpha=0.06,
                   color="#3b7dd8")
        dc = ctx[[col, "slack_l1"]].dropna()
        if len(dc):
            rc = spearmanr(dc[col], dc["slack_l1"]).statistic \
                if dc[col].nunique() > 1 else float("nan")
            ax.scatter(dc[col], dc["slack_l1"], s=2, alpha=0.06,
                       color="#d83b3b", label=f"contextual $\\rho$={rc:.3f}")
            ax.legend(markerscale=8, fontsize=8)
        ax.set_xlabel(lab); ax.set_ylabel("L1 slack")
        ax.set_title(f"all $\\rho$={rho:.3f}", fontsize=9)
    fig.suptitle("Witnesses vs LP slack (F4 redundancy scan)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_witness_vs_slack.png", dpi=150)
    plt.close(fig)

    # ---- fig 3: cf1_soft margin vs slack linear identity (contextual only)
    d = df[(~df["lp_feasible"].astype(bool)) &
           (~df["borderline_band"].astype(bool))]
    x = 1.0 - d["cf1_soft"].to_numpy()
    y = d["slack_l1"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    A_ = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
    pred = A_ @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(x, y, s=2, alpha=0.08)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, coef[0] * xs + coef[1], "r-",
            label=f"fit: slack={coef[0]:.3f}*(1-t*)+{coef[1]:.2e}, R²={r2:.5f}")
    ax.set_xlabel("1 - cf1_soft"); ax.set_ylabel("L1 slack")
    ax.legend(fontsize=8)
    ax.set_title("Population-level redundancy of cf1_soft with LP slack\n"
                 "(contextual instances)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_cf1soft_identity.png", dpi=150)
    plt.close(fig)

    # ---- fig 4: hierarchy placement (possibilistic tier) incl. sparse stream
    counts = hier["cf1_hard"].value_counts().sort_index()
    bysrc = []
    for src in ["uniform", "boundary", "sparse"]:
        ids = set(df[df["source"] == src]["instance_id"])
        h = hier[hier["instance_id"].isin(ids)]
        bysrc.append((src, len(h),
                      int((h["cf1_hard"] < 1.0).sum()) if len(h) else 0))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    counts.plot(kind="bar", ax=axes[0], color="#3b7dd8")
    axes[0].set_title("cf1_hard distribution (all contextual)")
    axes[0].set_xlabel("cf1_hard")
    labels = [b[0] for b in bysrc]
    tot = [b[1] for b in bysrc]
    strong = [b[2] for b in bysrc]
    xpos = np.arange(len(bysrc))
    axes[1].bar(xpos - 0.18, tot, width=0.36, label="contextual",
                color="#3b7dd8")
    axes[1].bar(xpos + 0.18, strong, width=0.36, label="strongly contextual",
                color="#d83b3b")
    axes[1].set_xticks(xpos, labels)
    axes[1].set_yscale("symlog")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Possibilistic tier fires only on sparse supports")
    fig.tight_layout()
    fig.savefig(FIG / "fig_hierarchy_placement.png", dpi=150)
    plt.close(fig)

    # ---- fig 5: KL non-redundancy (log-log within contextual)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    d = df[(~df["lp_feasible"].astype(bool)) &
           (~df["borderline_band"].astype(bool))]
    d = d[(d["slack_l1"] > 0) & (d["kl_contextuality"] > 0)]
    ax.scatter(d["slack_l1"], d["kl_contextuality"], s=2, alpha=0.08)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L1 slack"); ax.set_ylabel("KL contextuality")
    ax.set_title("KL is not a monotone transform of LP slack")
    fig.tight_layout()
    fig.savefig(FIG / "fig_kl_nonredundancy.png", dpi=150)
    plt.close(fig)

    print("figures written to", FIG)


if __name__ == "__main__":
    main()
