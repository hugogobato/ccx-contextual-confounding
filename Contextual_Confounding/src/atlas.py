"""WP 1.4 structure atlas (rewritten to per-structure methods).

A1_iv        reference: numbers come from the Phase 1 main run (conditional form).
A2_iv_direct conditional-form response map (32-dim), full witness suite.
A3_proxy_collider MILP feasibility on reduced density; witnesses skipped
             (feasible set is a finite union of polytopes, not convex).
A4_bow       generic joint form, full suite.
"""
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

from models import (ATLAS_SPECS, NM_EQUALITY_FLAGS, build_conditional_map_A2,  # noqa: E402
                    build_u_explicit_map, discover_facets, classify_trivial,
                    collider_feasible_milp, ROOT)
from witnesses import slack_and_feasible, cf1_soft, degree_signed, kl_em_batch  # noqa: E402
from dgps import uniform_conditionals  # noqa: E402

RES = ROOT / "results" / "phase1_enumeration"
RAW = ROOT / "results" / "raw" / "atlas"


def g2_ZY(e_joint):
    """G^2 for marginal independence Z vs Y in layout row = 4z+2x+y."""
    p = np.asarray(e_joint, float)
    p = p / p.sum()
    pz0 = p[0:4].sum()
    stat = 0.0
    for y in (0, 1):
        py = p[[i for i in range(8) if i % 2 == y]].sum()
        for z in (0, 1):
            pzy = sum(p[4 * z + 2 * x + y] for x in range(2))
            pz = p[4 * z:4 * z + 4].sum()
            if pzy > 0 and pz > 0:
                stat += pzy * np.log(pzy / (pz * py))
    return 2.0 * float(stat)


def suite(E, source, name, Mf, H, b):
    fac_ok = np.all(H.astype(float) @ E.T <= b[:, None].astype(float) + 1e-9,
                    axis=0)
    kl_vals, _ = kl_em_batch(Mf, E)
    rows = []
    for i in range(len(E)):
        e = E[i]
        slack, feas = slack_and_feasible(Mf, e)
        borderline = 1e-9 < slack < 1e-7
        rows.append({
            "structure": name, "instance_id": f"{name}_{source}_{i}",
            "source": source,
            **{f"e_{j}": float(e[j]) for j in range(len(e))},
            "lp_feasible": bool(feas), "facets_ok": bool(fac_ok[i]),
            "agree_c1a": bool(feas == bool(fac_ok[i])),
            "slack_l1": float(slack), "borderline_band": bool(borderline),
            "cf1_soft": cf1_soft(Mf, e),
            "degree_signed": degree_signed(Mf, e),
            "kl_contextuality": float(kl_vals[i]),
            "g2_ZY": g2_ZY(e) if name == "A3_proxy_collider" else np.nan,
        })
    return rows


def run_structure(name, spec, order_idx, seeds_cfg):
    t0 = time.time()
    method = spec.get("method", "joint")
    rng = np.random.default_rng(seeds_cfg["atlas"]["seed_start"] + 1000 * order_idx)

    if method == "reference":
        return None, None  # supplied by Phase 1 main run

    if method == "conditional":
        M = build_conditional_map_A2()
        D, Q = M.shape
        n_u = seeds_cfg["atlas"]["uniform_per_structure"]
        E_u = uniform_conditionals(rng, n_u)          # two Dirichlet blocks
        H, b = discover_facets(M, seed=0)
        trivial = classify_trivial(H, b, M)
        rows = suite(E_u, "uniform", name, M.astype(float), H, b)
        n_b = seeds_cfg["atlas"]["near_facet_per_structure"]
        ef = rng.dirichlet(np.ones(Q), size=n_b) @ M.T.astype(float)
        ed = uniform_conditionals(rng, n_b)
        alpha = rng.uniform(0.02, 0.3, size=(n_b, 1))
        E_b = (1 - alpha) * ef + alpha * ed   # near-polytope mixtures
        rows += suite(E_b, "boundary", name, M.astype(float), H, b)

    elif method == "joint":
        M = build_u_explicit_map(spec)
        D, Q = M.shape
        n_u = seeds_cfg["atlas"]["uniform_per_structure"]
        E_u = rng.dirichlet(np.ones(D), size=n_u)
        H, b = discover_facets(M, seed=0)
        trivial = classify_trivial(H, b, M)
        rows = suite(E_u, "uniform", name, M.astype(float), H, b)
        n_b = seeds_cfg["atlas"]["near_facet_per_structure"]
        ef = rng.dirichlet(np.ones(Q), size=n_b) @ M.T.astype(float)
        ed = rng.dirichlet(np.ones(D), size=n_b)
        alpha = rng.uniform(0.02, 0.3, size=(n_b, 1))
        E_b = (1 - alpha) * ef + alpha * ed
        rows += suite(E_b, "boundary", name, M.astype(float), H, b)

    elif method == "milp":
        n_u = min(seeds_cfg["atlas"]["uniform_per_structure"], 1600)
        E_u = rng.dirichlet(np.ones(8), size=n_u)
        rows = []
        for i, e in enumerate(E_u):
            pz = e[0:4].sum(), e[4:8].sum()
            feas = collider_feasible_milp(e) if min(pz) > 1e-12 else False
            fn_null = (i % 2 == 0)
            if fn_null and min(pz) > 1e-12:
                # factorized null p(z)p(x|z)p(y): satisfies the implied
                # equality Z indep Y exactly by construction
                a = rng.dirichlet(np.ones(2))
                px = [rng.dirichlet(np.ones(2)) for _ in range(2)]
                c = rng.dirichlet(np.ones(2))
                e = np.zeros(8)
                for z in range(2):
                    for x in range(2):
                        for y in range(2):
                            e[4 * z + 2 * x + y] = (a[z] * px[z][x]
                                                    * c[y])
            if fn_null:
                feas = collider_feasible_milp(e)
            rows.append({
                "structure": name,
                "instance_id": f"{name}_{'factorized_null' if fn_null else 'uniform'}_{i}",
                "source": "factorized_null" if fn_null else "uniform",
                **{f"e_{j}": float(e[j]) for j in range(8)},
                "lp_feasible": bool(feas), "facets_ok": np.nan,
                "agree_c1a": np.nan, "slack_l1": np.nan,
                "borderline_band": False, "cf1_soft": np.nan,
                "degree_signed": np.nan, "kl_contextuality": np.nan,
                "g2_ZY": g2_ZY(e),
            })
        df = pd.DataFrame(rows)
        RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW / f"{name}.csv", index=False)
        ctx = df[~df["lp_feasible"]]
        strict_ctx = ctx[ctx["source"] == "factorized_null"]
        summary = {
            "structure": name, "description": spec["desc"],
            "n_response_coords": 8, "n_observed_coords": 8,
            "n_facets_total": np.nan,
            "n_facets_nontrivial": np.nan,
            "note_facets": "feasible set nonconvex (union of polytopes); no H-rep",
            "n_scanned": int(len(df)), "n_feasible": int(df["lp_feasible"].sum()),
            "n_contextual": int(len(ctx)),
            "frac_contextual": float(len(ctx) / max(len(df), 1)),
            "cf1_soft_min": np.nan, "cf1_soft_mean_margin": np.nan,
            "degree_mean": np.nan,
            "has_nm_equality_constraints": NM_EQUALITY_FLAGS[name][0],
            "nm_flag_note": NM_EQUALITY_FLAGS[name][1],
            "n_strict_equality_constrained": int(len(strict_ctx)),
            "agree_rate": np.nan,
            "runtime_s": round(time.time() - t0, 1),
        }
        print(json.dumps(summary, indent=2, default=str), flush=True)
        return df, summary

    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / f"{name}.csv", index=False)
    ctx = df[(~df["lp_feasible"]) & (~df["borderline_band"])]
    strict_ctx = ctx
    if name == "A3_proxy_collider":
        strict_ctx = ctx[ctx["g2_ZY"] < 1e-6] if len(ctx) else ctx
    summary = {
        "structure": name, "description": spec["desc"],
        "n_response_coords": M.shape[1], "n_observed_coords": M.shape[0],
        "n_facets_total": int(len(H)),
        "n_facets_nontrivial": int((~trivial).sum()),
        "n_scanned": int(len(df)), "n_feasible": int(df["lp_feasible"].sum()),
        "n_contextual": int(len(ctx)),
        "frac_contextual": float(len(ctx) / max(len(df), 1)),
        "cf1_soft_min": float(np.nanmin(1 - ctx["cf1_soft"])) if len(ctx) else np.nan,
        "cf1_soft_mean_margin": float(np.nanmean(1 - ctx["cf1_soft"])) if len(ctx) else np.nan,
        "degree_mean": float(np.nanmean(ctx["slack_l1"])) if len(ctx) else np.nan,
        "has_nm_equality_constraints": NM_EQUALITY_FLAGS[name][0],
        "nm_flag_note": NM_EQUALITY_FLAGS[name][1],
        "n_strict_equality_constrained": int(len(strict_ctx)),
        "agree_rate": float(df["agree_c1a"].mean()),
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return df, summary


def main(quick=False):
    seeds_cfg = json.loads((ROOT / "configs" / "seeds.json").read_text())
    if quick:
        seeds_cfg["atlas"] = {"uniform_per_structure": 400,
                              "near_facet_per_structure": 150,
                              "seed_start": 777000}
    summaries = []
    for order_idx, (name, spec) in enumerate(ATLAS_SPECS.items()):
        df, summ = run_structure(name, spec, order_idx, seeds_cfg)
        if summ is None:
            summaries.append({
                "structure": name, "description": spec["desc"],
                "note": "supplied by Phase 1 WP 1.3 main run",
                "has_nm_equality_constraints": NM_EQUALITY_FLAGS[name][0],
                "nm_flag_note": NM_EQUALITY_FLAGS[name][1],
            })
        else:
            summaries.append(summ)
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / "structure_atlas.csv"
    pd.DataFrame(summaries).to_csv(out, index=False)
    print("\natlas written:", out)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
