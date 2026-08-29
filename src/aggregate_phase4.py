"""Phase 4 aggregator: processed tables + predeclared Gate D evaluation.

Inputs:
- results/raw/phase4/p4d{null,alt,adv}_*.csv   (discrete ladder, local)
- results/raw/phase4/p4c{null,alt,adv}_*.csv   (continuous new cells)
- results/raw/phase3/wp32_*.csv, wp33_*.csv    (verified Gate C raw data;
  the Phase 4 continuous grid reuses these rows, D-P4.2)
- results/raw/phase4/{p4scale,p4realdata}.csv

Outputs (results/phase4_grid/):
- discrete_envelope_cvs.csv     envelope CVs (D6 rule) per (cell,n,tail,stat,engine,alpha)
- discrete_null_sizes.csv       adaptive size per (cell,n,tail,kind,stat,engine,alpha)
- discrete_power_curves.csv     power vs rho with null_size + advantage columns
- discrete_adversarial.csv      FPR of A1-A4 vs envelopes
- continuous_null_sizes.csv     pooled wp32 + p4cnull size table
- continuous_separation.csv     power vs b over the Phase 4 grid (wp33 + p4calt)
- continuous_adversarial.csv    FPR of B1-B4 (own-draw permutation CVs)
- scaling_phase4.csv            runtime/memory measurements
- realdata_smoke.csv            witness values + rejection flags (alpha 0.05)
- gateD_evaluation.csv          predeclared Gate D rule inputs per arm
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase4_grid"
RAW4 = ROOT / "results" / "raw" / "phase4"
RAW3 = ROOT / "results" / "raw" / "phase3"

ALPHAS = [round(0.01 * a, 2) for a in range(1, 21)]
PRIM_TRIM = {"k1": 0.05, "k2": 0.05, "hsic": 0.0}
P4 = json.loads((ROOT / "configs" / "phase4.json").read_text())
B_GRID = P4["continuous_grid"]["b_grid"]
RHO_GRID = P4["discrete_grid"]["rho_grid"]
MONO_TOL = -0.08          # per-panel monotonicity tolerance (as Gate C)
ADV_MIN = 0.20            # Gate D advantage threshold
SIZE_MAX = 0.10           # 2 * alpha at the binding level
FPR_MAX_GO = 0.10


def _read_glob(glob_dir, pattern):
    files = sorted(glob_dir.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) \
        if files else pd.DataFrame()


def _cv_cols(df):
    return [c for c in df.columns if c.startswith("cv_")]


# =============================================================== discrete

def discrete_blocks():
    dnull = _read_glob(RAW4, "p4dnull_*.csv")
    dalt = _read_glob(RAW4, "p4dalt_*.csv")
    dadv = _read_glob(RAW4, "p4dadv_*.csv")
    if not len(dnull):
        raise SystemExit("no p4dnull raw rows")

    # ---- adaptive size per null config (boot rows)
    cc = _cv_cols(dnull)
    boot = dnull[dnull["B"] > 0].copy()
    rej = pd.DataFrame({"cell": boot["cell"], "n": boot["n"],
                        "tail": boot["tail"], "kind": boot["kind"],
                        "stat": boot["stat"], "engine": boot["engine"]})
    for c in cc:
        rej[c] = boot["stat_obs"].to_numpy() > boot[c].to_numpy()
    size = rej.groupby(["cell", "n", "tail", "kind", "stat", "engine"],
                       as_index=False)[cc].mean()
    size["n_datasets"] = rej.groupby(
        ["cell", "n", "tail", "kind", "stat", "engine"])[cc[0]] \
        .count().to_numpy()
    size.rename(columns={c: f"size_{c[3:]}" for c in cc}, inplace=True)
    size.to_csv(RES / "discrete_null_sizes.csv", index=False)

    # ---- envelope CVs: max over null kinds of mean own-draw CV (D6)
    mean_cv = boot.groupby(["cell", "n", "tail", "stat", "engine"],
                           as_index=False)[cc].mean()
    env = mean_cv.groupby(["cell", "n", "tail", "stat", "engine"],
                          as_index=False)[cc].max()
    env.to_csv(RES / "discrete_envelope_cvs.csv", index=False)
    env_lut = {(r.cell, int(r.n), r.tail, r.stat, r.engine): r
               for r in env.itertuples()}

    def env_cv(cell, n, tail, stat, engine, a=0.05):
        r = env_lut.get((cell, int(n), tail, stat, engine))
        return getattr(r, f"cv_{a:.2f}") if r is not None else np.nan

    # ---- power curves (alt arm) vs envelopes
    alt = dalt.copy()
    a = 0.05
    alt["cv_env_05"] = [env_cv(r.cell, r.n, r.tail, r.stat, r.engine, a)
                        for r in alt.itertuples()]
    alt["rej"] = alt["stat_obs"] > alt["cv_env_05"]
    pw = alt.groupby(["cell", "tail", "rho", "n", "stat", "engine"],
                     as_index=False).agg(power=("rej", "mean"),
                                         n_reps=("rej", "size"))
    # matched null size: mean adaptive rejection rate at alpha 0.05 over
    # null kinds at the same (cell, n, tail)
    sz05 = rej[["cell", "n", "tail", "stat", "engine", "cv_0.05"]] \
        .rename(columns={"cv_0.05": "rej05"})
    sz05 = sz05.groupby(["cell", "n", "tail", "stat", "engine"],
                        as_index=False)["rej05"].mean()
    sz05 = sz05.rename(columns={"rej05": "null_size"})
    pw = pw.merge(sz05, on=["cell", "n", "tail", "stat", "engine"],
                  how="left")
    pw["advantage"] = pw["power"] - pw["null_size"]
    pw.to_csv(RES / "discrete_power_curves.csv", index=False)

    # ---- discrete adversarial FPR vs envelopes
    if len(dadv):
        dadv["cv_env_05"] = [env_cv(r.cell, r.n, r.tail_env, r.stat,
                                    r.engine, a)
                             for r in dadv.itertuples()]
        dadv["rej"] = dadv["stat_obs"] > dadv["cv_env_05"]
        adv = dadv.groupby(["adv_id", "cell", "tail_env", "stat", "engine"],
                           as_index=False).agg(
            fpr=("rej", "mean"), n_reps=("rej", "size"),
            feas_frac=("feas_contam", "mean"))
        adv.to_csv(RES / "discrete_adversarial.csv", index=False)
    return pw, sz05


# ============================================================= continuous

def continuous_blocks():
    cnull = _read_glob(RAW4, "p4cnull_*.csv")
    w32 = _read_glob(RAW3, "wp32_*.csv")
    # v4 (D-P4.9): matched nulls for the full grid come from p4cnull;
    # WP3.2 rows are a different (linear-detrend) statistic version.
    call = cnull
    cc = _cv_cols(call)
    size = call.groupby(["kind", "noise", "n", "d", "method", "trim"],
                        as_index=False).agg(
        size_05=("stat_obs", "first"), n_datasets=("stat_obs", "size"))
    # proper size: rejection vs own-draw CVs
    call2 = call.copy()
    call2["rej"] = call2["stat_obs"] > call2["cv_0.05"]
    size = call2.groupby(["kind", "noise", "n", "d", "method", "trim"],
                         as_index=False).agg(
        size_05=("rej", "mean"), n_datasets=("rej", "size"))
    size.to_csv(RES / "continuous_null_sizes.csv", index=False)

    # ---- CV transfer table (primary trim per method; nearest-n fallback)
    prim = call2[call2.apply(
        lambda r: r["trim"] == PRIM_TRIM.get(r["method"], r["trim"]),
        axis=1)]
    cvt = prim.groupby(["kind", "noise", "n", "d", "method"],
                       as_index=False).agg(cv_05=("cv_0.05", "mean"))
    lut = {(r.kind, r.noise, int(r.n), int(r.d), r.method): r.cv_05
           for r in cvt.itertuples()}
    ns_by = {}
    for (k, nz, n, d, m) in lut:
        ns_by.setdefault((nz, d, m), set()).add(n)

    def cv_for(kind, noise, n, d, meth):
        if (kind, noise, int(n), int(d), meth) in lut:
            return lut[(kind, noise, int(n), int(d), meth)]
        cand = ns_by.get((noise, d, meth), set())
        if not cand:
            return np.nan
        smaller = [m for m in sorted(cand) if m <= n]
        chosen = max(smaller) if smaller else min(cand)
        return lut.get((kind, noise, chosen, d, meth), np.nan)

    # ---- separation: wp33 rows + p4calt rows, Phase 4 b grid
    # v4 (D-P4.9): WP3.3 reuse retired; the full continuous grid comes
    # from p4calt (quadratic detrend) with matched p4cnull CVs.
    sep_raw = _read_glob(RAW4, "p4calt_*.csv")
    sep_raw = sep_raw[sep_raw["b"].isin(B_GRID) &
                      sep_raw["n"].isin(P4["continuous_grid"]["n_grid"]) &
                      sep_raw["noise"].isin(P4["continuous_grid"]["noise"])]
    kind_map = {"conf_lin": "null_gauss", "conf_nonlin": "null_nonparam"}
    sep_raw["cv_0.05"] = [cv_for(kind_map[r["kind"]], r["noise"], r["n"],
                                 r["d"], r["method"])
                          for _, r in sep_raw.iterrows()]
    sep_raw["rej"] = sep_raw["stat_obs"] > sep_raw["cv_0.05"]
    sep = sep_raw.groupby(["kind", "noise", "n", "d", "b", "method"],
                          as_index=False).agg(power=("rej", "mean"),
                                              n_reps=("rej", "size"))
    sz_lu = {(r.noise, int(r.n), int(r.d), r.method): r.size_05
             for r in size.itertuples()
             if r.trim == PRIM_TRIM.get(r.method, 0.05)}
    sep["null_size"] = [sz_lu.get((r.noise, int(r.n), int(r.d), r.method),
                                  np.nan) for r in sep.itertuples()]
    sep["advantage"] = sep["power"] - sep["null_size"]
    # hsic power on the same cell for the baseline-advantage computation
    hs = sep[sep["method"] == "hsic"][["kind", "noise", "n", "d", "b",
                                       "power"]] \
        .rename(columns={"power": "power_hsic"})
    sep = sep.merge(hs, on=["kind", "noise", "n", "d", "b"], how="left")
    sep.to_csv(RES / "continuous_separation.csv", index=False)

    # ---- continuous adversarial FPR (own-draw permutation CVs)
    cadv = _read_glob(RAW4, "p4cadv_*.csv")
    if len(cadv):
        cadv["rej"] = cadv["stat_obs"] > cadv["cv_0.05"]
        cadv.groupby(["adv_id", "n", "d", "noise", "method"],
                     as_index=False).agg(fpr=("rej", "mean"),
                                         n_reps=("rej", "size")) \
            .to_csv(RES / "continuous_adversarial.csv", index=False)
    return sep, cadv


# ============================================================== scaling

def scaling_block():
    sc = _read_glob(RAW4, "p4scale.csv")
    if len(sc):
        sc.to_csv(RES / "scaling_phase4.csv", index=False)
    return sc


# ============================================================ real data

def realdata_block(env_lut):
    rd = _read_glob(RAW4, "p4realdata.csv")
    if not len(rd):
        return rd
    cc = _cv_cols(rd)
    for c in cc:
        rd[c] = pd.to_numeric(rd[c], errors="coerce")
    rd["reject_05_own"] = np.where(
        rd["cv_0.05"].notna(), rd["stat_obs"] > rd["cv_0.05"], np.nan)

    def env_join(r):
        cell = getattr(r, "cell", np.nan)
        if not (isinstance(cell, str) and cell):
            return np.nan
        if np.isfinite(getattr(r, "cv_0.05", np.nan)):
            return np.nan                      # own-draw CV already present
        e = env_lut.get((cell, int(r.n_env), "none", r.method, "none"))
        return getattr(e, "cv_0.05", np.nan) if e is not None else np.nan

    rd["cv_env_05"] = [env_join(r) for r in rd.itertuples()]
    rd["reject_05_env"] = rd["stat_obs"] > rd["cv_env_05"]
    rd.to_csv(RES / "realdata_smoke.csv", index=False)
    return rd


# ============================================================= Gate D

def gate_d(dpw, cont_sep, dcadv, ccadv):
    rows = []

    # ---------- continuous arm: witness k1/k2 vs strongest baseline hsic
    fpr_c, fpr_c_ins, fpr_cfg = {}, {}, {}
    if len(ccadv):
        g = ccadv.copy()
        g["rej"] = g["stat_obs"] > g["cv_0.05"]
        per_cfg = g.groupby(["method", "adv_id"])["rej"].mean()
        fpr_c = per_cfg.groupby("method").max().to_dict()   # worst config
        fpr_cfg = {m: per_cfg[m].to_dict() for m in
                   per_cfg.index.get_level_values(0).unique()}
        # in-pipeline-class configs (B4's strong curvature is outside the
        # linear-detrend surrogate's reach; see gate_D_memo L4)
        ins = per_cfg.reset_index()
        ins = ins[~ins["adv_id"].isin(["B4_heavy_curvature"])]
        fpr_c_ins = ins.groupby("method")["rej"].max().to_dict()
    for meth in ("k1", "k2"):
        sub = cont_sep[cont_sep["method"] == meth].copy()
        if not len(sub):
            continue
        sub["adv_vs_hsic"] = sub["power"] - sub["power_hsic"]
        sub["size_ok"] = sub["null_size"] <= SIZE_MAX
        sub["cell_ok"] = (sub["adv_vs_hsic"] >= ADV_MIN) & sub["size_ok"]
        qual_cells = 0
        total_cells = 0
        mono_all = True
        for (kd, nz, dv, nv), g in sub.groupby(["kind", "noise", "d", "n"]):
            g = g.sort_values("b")
            total_cells += len(g)
            qual_cells += int(g["cell_ok"].sum())
            if not np.all(np.diff(g["power"].to_numpy()) >= MONO_TOL):
                mono_all = False
        share = qual_cells / max(total_cells, 1)
        rows.append({
            "arm": "continuous", "method": meth, "baseline": "hsic",
            "adv_threshold": ADV_MIN, "share_ge_020": bool(share >= 0.20),
            "region_share": share, "qualifying_cells": qual_cells,
            "grid_cells": total_cells, "monotone_all_panels": mono_all,
            "adversarial_fpr": float(fpr_c.get(meth, np.nan)),
            "adversarial_fpr_in_class": float(fpr_c_ins.get(meth, np.nan)),
            "adversarial_fpr_by_config": json.dumps(
                fpr_cfg.get(meth, {})),
            "gateD_go_input": bool(share >= 0.20 and
                                   np.isfinite(fpr_c.get(meth, np.nan)) and
                                   fpr_c.get(meth, 1.0) <= FPR_MAX_GO and
                                   mono_all),
        })

    # ---------- discrete arm: best witness stat vs best LP baseline
    fpr_d = np.nan
    if len(dcadv):
        # worst FPR over ALL adversarial configs (A1-A3 are pure nulls;
        # A4's contaminated law classified infeasible at population level
        # in every replication, i.e. it is a detection arm, and its rows
        # are excluded from the false-positive rate)
        pure = dcadv[(dcadv["adv_id"] != "A4_feasible_contamination") |
                     (dcadv["feas_frac"] > 0)]
        fpr_d = float(pure["fpr"].max()) if len(pure) else np.nan
    cells = ["-".join(map(str, c))
             for c in P4["discrete_grid"]["cells"]]
    tails = P4["discrete_grid"]["tails"]
    ns = P4["discrete_grid"]["n_grid"]
    w_stats = ["kl_plugin", "kl_split", "kl_crossfit"]
    b_stats = ["cf1_margin", "slack_plugin"]
    engs = ["para_boot", "crt_cond"]

    def best_pair(pw, cell, tail, n, rho, pairs):
        best = (np.nan, None, None)
        for s, e in pairs:
            r = pw[(pw["cell"] == cell) & (pw["tail"] == tail) &
                   (pw["n"] == n) & (pw["rho"] == rho) &
                   (pw["stat"] == s) & (pw["engine"] == e)]
            if len(r) and np.isfinite(r["power"].iloc[0]) and \
                    (np.isnan(best[0]) or r["power"].iloc[0] > best[0]):
                best = (float(r["power"].iloc[0]), s, e)
        return best

    # witness variant is restricted to size-eligible (stat, engine) pairs;
    # selection by power among size-controlled variants only
    variant_ok = {}
    for s in w_stats:
        for e in engs:
            sub = dpw[(dpw["stat"] == s) & (dpw["engine"] == e)]
            variant_ok[(s, e)] = bool(
                len(sub) and (sub["null_size"] <= SIZE_MAX).all())
    elig = [k for k, v in variant_ok.items() if v] or \
        [(s, e) for s in w_stats for e in engs]
    base_pairs = [(s, e) for s in b_stats for e in engs]

    qual_d = 0
    total_d = 0
    for cell in cells:
        for tail in tails:
            for n in ns:
                for rho in RHO_GRID:
                    total_d += 1
                    pw_v, s_v, e_v = best_pair(dpw, cell, tail, n, rho,
                                               elig)
                    bl_v, bl_s, bl_e = best_pair(dpw, cell, tail, n, rho,
                                                 base_pairs)
                    if s_v is None or bl_s is None:
                        continue
                    sz = dpw[(dpw["cell"] == cell) & (dpw["tail"] == tail) &
                             (dpw["n"] == n) & (dpw["rho"] == rho) &
                             (dpw["stat"] == s_v) &
                             (dpw["engine"] == e_v)]["null_size"]
                    size_ok = bool(len(sz)) and sz.iloc[0] <= SIZE_MAX
                    if (pw_v - bl_v) >= ADV_MIN and size_ok:
                        qual_d += 1
    share_d = qual_d / max(total_d, 1)
    rows.append({
        "arm": "discrete",
        "method": "kl_family(best_size_eligible)",
        "baseline": "calibrated_LP(cf1/slack best)",
        "adv_threshold": ADV_MIN, "share_ge_020": bool(share_d >= 0.20),
        "region_share": share_d, "qualifying_cells": qual_d,
        "grid_cells": total_d,
        "monotone_all_panels": "",
        "adversarial_fpr": fpr_d,
        "gateD_go_input": bool(share_d >= 0.20 and
                               np.isfinite(fpr_d) and
                               fpr_d <= FPR_MAX_GO),
    })
    gd = pd.DataFrame(rows)
    gd.to_csv(RES / "gateD_evaluation.csv", index=False)
    return gd


def main():
    RES.mkdir(parents=True, exist_ok=True)
    dpw, _sz05 = discrete_blocks()
    cont_sep, ccadv = continuous_blocks()
    scaling_block()
    env = pd.read_csv(RES / "discrete_envelope_cvs.csv")
    env_lut = {(r.cell, int(r.n), r.tail, r.stat, r.engine): r
               for r in env.itertuples()}
    realdata_block(env_lut)
    dcadv = pd.read_csv(RES / "discrete_adversarial.csv") \
        if (RES / "discrete_adversarial.csv").exists() else pd.DataFrame()
    gd = gate_d(dpw, cont_sep, dcadv, ccadv)
    print(gd.to_string(index=False))


if __name__ == "__main__":
    main()
