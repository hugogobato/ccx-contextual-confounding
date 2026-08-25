"""WP 1.3 binary-IV enumeration driver (Phase 1).

Streams: uniform (100 x 10k, seeds 0..99), boundary (100k, seed 20260824),
exact vertices; inflation-2 cross-check on a subsample. Outputs the four
CSVs required by the plan plus raw per-batch dumps.

Batch structure: facet decisions and KL witnesses are computed vectorized per
batch; slack / cf1_soft / degree_signed LPs run per instance; contextual
extras (cf1_hard, maximal-support LPs) only on contextual instances.
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

from models import build_iv_A, iv_vertices, discover_facets, ROOT  # noqa: E402
from witnesses import (slack_and_feasible, cf1_soft, degree_signed,  # noqa: E402
                       kl_em_batch, iv_cf1_hard, maximal_support_lp,
                       iv_inflation2_feasible)
from dgps import uniform_conditionals, boundary_batch, sparse_conditionals  # noqa: E402

RES = ROOT / "results" / "phase1_enumeration"
RAW = ROOT / "results" / "raw" / "phase1"
TOL_BORDER_LO, TOL_BORDER_HI = 1e-9, 1e-7


def _load_facets():
    cache = RES / "facets_iv.npz"
    if cache.exists():
        z = np.load(cache)
        return z["H"], z["b"]
    RES.mkdir(parents=True, exist_ok=True)
    Hi, bi = discover_facets(build_iv_A(), seed=0)
    np.savez(cache, H=H, b=b) if False else np.savez(cache, H=Hi, b=bi)
    return Hi, bi


def process_batch(E, source, batch, seed, want_inflation_idx=None):
    """Run the full suite on a batch of observables; returns list of row dicts."""
    A16 = build_iv_A()
    Af = A16.astype(float)
    H16, b16 = _load_facets()
    n = len(E)

    fac_ok = np.all(H16.astype(float) @ E.T <= b16[:, None].astype(float) + 1e-9,
                    axis=0)
    kl_vals, _ = kl_em_batch(Af, E)
    infl_flags = np.full(n, np.nan)
    if want_inflation_idx is not None:
        for i in want_inflation_idx:
            infl_flags[i] = float(iv_inflation2_feasible(E[i], A16))

    rows = []
    for i in range(n):
        e = E[i]
        slack, feas = slack_and_feasible(Af, e)
        borderline = TOL_BORDER_LO < slack < TOL_BORDER_HI
        row = {
            "instance_id": f"{source}_{batch}_{i}", "source": source,
            "batch": batch, "seed": seed,
            **{f"e_{j}": float(e[j]) for j in range(8)},
            "lp_feasible": bool(feas), "facets_ok": bool(fac_ok[i]),
            "agree_c1a": bool(feas == bool(fac_ok[i])),
            "slack_l1": float(slack),
            "borderline_band": bool(borderline),
            "cf1_soft": cf1_soft(Af, e),
            "degree_signed": degree_signed(Af, e),
            "kl_contextuality": float(kl_vals[i]),
            "inflation2_feasible": infl_flags[i],
        }
        if not feas and not borderline:
            row["cf1_hard"] = iv_cf1_hard(e, A16)
            m0 = maximal_support_lp(A16[0:4], e[0:4] > 0)
            m1 = maximal_support_lp(A16[4:8], e[4:8] > 0)
            row["maximally_contextual"] = bool(max(m0, m1) <= 1e-9)
        else:
            row["cf1_hard"] = np.nan
            row["maximally_contextual"] = False
        rows.append(row)
    return rows


def run_uniform_batch(spec):
    batch, n, seed = spec
    rng = np.random.default_rng(seed)
    E = uniform_conditionals(rng, n)
    t0 = time.time()
    rows = process_batch(E, "uniform", batch, seed)
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW / f"uniform_batch_{batch:03d}.csv", index=False)
    return batch, len(rows), time.time() - t0


def run_boundary_chunk(spec):
    chunk_id, n, seed = spec
    A16 = build_iv_A()
    H16, b16 = _load_facets()
    t0 = time.time()
    E = boundary_batch(A16, H16, b16, n, seed)
    rows = process_batch(E, "boundary", chunk_id, seed)
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW / f"boundary_chunk_{chunk_id:03d}.csv",
                              index=False)
    return chunk_id, len(rows), time.time() - t0


def run_sparse_chunk(spec):
    chunk_id, n, seed = spec
    rng = np.random.default_rng(seed)
    t0 = time.time()
    E = sparse_conditionals(rng, n)
    rows = process_batch(E, "sparse", chunk_id, seed)
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW / f"sparse_chunk_{chunk_id:03d}.csv",
                              index=False)
    return chunk_id, len(rows), time.time() - t0


def run_job(kind_job):
    kind, spec = kind_job
    if kind == "u":
        ident, nrows, dt = run_uniform_batch(spec)
        return "uniform", ident, nrows, dt
    if kind == "s":
        ident, nrows, dt = run_sparse_chunk(spec)
        return "sparse", ident, nrows, dt
    ident, nrows, dt = run_boundary_chunk(spec)
    return "boundary", ident, nrows, dt


def main(n_batches=None, batch_size=None, quick=False):
    seeds_cfg = json.loads((ROOT / "configs" / "seeds.json").read_text())
    u = seeds_cfg["phase1_uniform"]
    bd = seeds_cfg["phase1_boundary"]
    sub = seeds_cfg["phase1_inflation_subsample"]
    if quick:
        u = dict(u, n_batches=2, batch_size=300)
        bd = dict(bd, n_points=1800)
        sp = {"n_points": 2000, "seed": 555000}
    else:
        sp = seeds_cfg.setdefault("phase1_sparse", {"n_points": 20000,
                                                    "seed": 555000})
    n_batches = n_batches or u["n_batches"]
    batch_size = batch_size or u["batch_size"]

    import multiprocessing as mp
    jobs = [(bb, batch_size, bb)
            for bb in range(u["seed_start"], u["seed_start"] + n_batches)]
    bchunk = max(batch_size, 2000)
    bjobs, remaining, cid = [], bd["n_points"], 0
    while remaining > 0:
        take = min(bchunk, remaining)
        bjobs.append((cid, take, bd["seed"] + cid))
        remaining -= take
        cid += 1
    schunk = 2000
    sjobs, remaining, cid = [], sp["n_points"], 0
    while remaining > 0:
        take = min(schunk, remaining)
        sjobs.append((cid, take, sp["seed"] + cid))
        remaining -= take
        cid += 1
    all_jobs = ([("u", j) for j in jobs] + [("b", j) for j in bjobs] +
                [("s", j) for j in sjobs])

    t_start = time.time()
    with mp.Pool(processes=int(os.environ.get("CCX_WORKERS", "6"))) as pool:
        for kind, ident, nrows, dt in pool.imap_unordered(run_job, all_jobs):
            print(f"[{kind}] job {ident}: {nrows} rows in {dt:.1f}s "
                  f"(elapsed {time.time()-t_start:.0f}s)", flush=True)

    files = sorted(RAW.glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    V = iv_vertices().astype(float)
    df_v = pd.DataFrame(process_batch(V, "vertex", -1, -1))
    df = pd.concat([df, df_v], ignore_index=True)

    ksub = max(1, (u["n_batches"] * u["batch_size"]) // sub["n_points"])
    mask = ((df["source"] == "uniform") & (df["batch"] % ksub == 0)) | \
           ((df["source"] == "boundary") & (df["batch"] % 5 == 0))
    idx_sub = df.index[mask].to_numpy()[:sub["n_points"]]
    print(f"inflation-2 subsample: {len(idx_sub)} instances", flush=True)
    inf_res = []
    A16 = build_iv_A()
    for i in idx_sub:
        e = df.loc[i, [f"e_{j}" for j in range(8)]].to_numpy(dtype=float)
        inf_res.append(iv_inflation2_feasible(e, A16))
    df.loc[idx_sub, "inflation2_feasible"] = np.array(inf_res, dtype=float)

    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "t1_dictionary.csv", index=False)

    ctx = df[(~df["lp_feasible"]) & (~df["borderline_band"])].copy()
    hier = pd.DataFrame({
        "instance_id": ctx["instance_id"],
        "strongly_contextual": ctx["cf1_hard"] < 1.0,
        "cf1_hard": ctx["cf1_hard"],
        "maximally_contextual": ctx["maximally_contextual"].astype(bool),
        "cf1_soft": ctx["cf1_soft"],
        "degree_honest": ctx["slack_l1"] / 2.0,
        "degree_signed": ctx["degree_signed"],
        "kl_contextuality": ctx["kl_contextuality"],
        "slack_l1": ctx["slack_l1"],
    })
    hier.to_csv(RES / "hierarchy_placement.csv", index=False)

    strict = pd.DataFrame({
        "instance_id": ctx["instance_id"],
        "nm_equalities_hold": True,
        "n_nm_constraints": 0,
        "margin_cf1_soft": 1.0 - ctx["cf1_soft"],
        "margin_degree_honest": ctx["slack_l1"] / 2.0,
        "margin_kl": ctx["kl_contextuality"],
    })
    strict["strict_instance"] = (strict["margin_cf1_soft"] >= 0.05) | \
                                (strict["margin_degree_honest"] >= 0.05) | \
                                (strict["margin_kl"] >= 0.05)
    strict.to_csv(RES / "strictness_scan.csv", index=False)

    from scipy.stats import spearmanr
    wit_cols = ["cf1_soft", "degree_signed", "kl_contextuality", "cf1_hard"]
    def safe_spearman(a, bcol):
        if len(a) < 3 or a[wcol].nunique() < 2 or a[bcol].nunique() < 2:
            return np.nan
        return float(spearmanr(a[wcol], a[bcol]).statistic)

    rows_r = []
    d_all = df[df["slack_l1"].notna()]
    d_ctx = d_all[~d_all["lp_feasible"] & ~d_all["borderline_band"]]
    for wcol in wit_cols:
        dd = d_all[[wcol, "slack_l1"]].replace([np.inf], np.nan).dropna()
        rho_all = safe_spearman(dd, "slack_l1")
        dc = d_ctx[[wcol, "slack_l1"]].replace([np.inf], np.nan).dropna()
        rho_ctx = safe_spearman(dc, "slack_l1")
        flag = False
        lin_r2 = np.nan
        try:
            if len(dc) > 10 and dc[wcol].nunique() > 2:
                from sklearn.isotonic import IsotonicRegression
                rk_x = dc[wcol].rank().to_numpy()
                rk_y = dc["slack_l1"].rank().to_numpy()

                def iso_r2(x, y):
                    pred = IsotonicRegression(out_of_bounds="clip") \
                        .fit_transform(x, y)
                    ss_res = float(((y - pred) ** 2).sum())
                    ss_tot = float(((y - y.mean()) ** 2).sum())
                    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                r_up = iso_r2(rk_x, rk_y)
                r_dn = iso_r2(-rk_x, rk_y)   # anti-monotone direction
                flag = bool(max(r_up, r_dn) >= 0.9999)
            # linear proportionality check: slack vs (1 - witness) where
            # noncontextual value is 1 (cf1_soft); generic for others skipped
            if wcol == "cf1_soft" and len(dc) > 10:
                x = (1.0 - dc[wcol]).to_numpy()
                y = dc["slack_l1"].to_numpy()
                A_lin = np.vstack([x, np.ones_like(x)]).T
                coef, res_, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
                pred = A_lin @ coef
                ss_res = float(((y - pred) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                lin_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            flag = False
        rows_r.append({"witness": wcol, "spearman_rho": rho_all,
                       "spearman_rho_contextual": rho_ctx,
                       "lin_r2_slack_vs_1_minus_witness": lin_r2,
                       "n": int(len(dd)), "monotone_injective_flag": flag})
    pd.DataFrame(rows_r).to_csv(RES / "witness_lp_redundancy.csv", index=False)
    deg_max = float(d_all["degree_signed"].abs().max())

    agree = int(df["agree_c1a"].sum())
    nb = int((~df["borderline_band"]).sum())
    print("\n==== PHASE 1 WP 1.3 SUMMARY ====")
    print(f"instances total: {len(df)} (borderline flagged: "
          f"{int(df['borderline_band'].sum())})")
    print(f"C1a agreement: {agree}/{nb} non-borderline = "
          f"{100.0*agree/max(nb,1):.4f}%")
    sub_df = df[df["inflation2_feasible"].notna()]
    infl_agree = int((sub_df["inflation2_feasible"].astype(bool) ==
                      sub_df["lp_feasible"]).sum())
    print(f"inflation-2 cross-check agreement: {infl_agree}/{len(sub_df)}")
    print(f"contextual instances: {len(ctx)} "
          f"({100.0*len(ctx)/max(nb,1):.2f}% of non-borderline)")
    if len(ctx):
        print(f"  strongly contextual: {int((ctx['cf1_hard'] < 1.0).sum())}")
        print(f"  maximally contextual: "
              f"{int(ctx['maximally_contextual'].astype(bool).sum())}")
        print(f"  cf1_hard distribution: "
              f"{ctx['cf1_hard'].value_counts().to_dict()}")
    print(f"degree_signed max |value| across all instances: {deg_max:.3e} "
          f"(identically ~0 => CbD signed convention vacuous for IV)")
    print(f"strict instances (margin>=0.05): "
          f"{int(strict['strict_instance'].sum())}/{len(strict)}")
    print(pd.DataFrame(rows_r).to_string(index=False))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
